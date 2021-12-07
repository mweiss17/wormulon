import io
import os
import uuid
import time
from datetime import datetime
from wormulon.utils import NotAvailable, ExceptionInJob, JobFailure, JobTimeout, serialize, JobState, dump_yaml
from dataclasses import dataclass
from wormulon.tpu.fncall import FunctionCall
from wormulon.tpu.tpu_job import TPUJob
from wormulon.tpu.bucket import Bucket
from typing import Union, Any, Optional


@dataclass
class TPUJobHandler(object):
    # Required at instantiation
    experiment_directory: str
    bucket: Bucket
    tpu_job: TPUJob
    function_call: FunctionCall
    # Can be filled in later
    job_output: Union[Any, NotAvailable] = NotAvailable()
    has_timed_out: bool = False
    job_has_died: bool = False
    last_heartbeat: datetime = None

    @classmethod
    def instantiate(
        cls, bucket: Bucket, exp_dir, fn, trainstate, job_kwargs
    ):
        try:
            trainstate = bucket.get_latest_trainstate(exp_dir)
        except IndexError:
            print("No trainstate found in bucket.")
        tpu_job = TPUJob(**job_kwargs)
        function_call = FunctionCall(fn, trainstate, job_kwargs)
        return cls(
            bucket=bucket,
            experiment_directory=exp_dir,
            function_call=function_call,
            tpu_job=tpu_job,
        )

    @property
    def working_directory(self):
        path = os.path.join(self.experiment_directory, self.tpu_job.job_id)
        return path

    @property
    def function_call_serialization_path(self):
        return os.path.join(self.working_directory, "function_call.pkl")

    @property
    def function_call_configuration_path(self):
        return os.path.join(self.working_directory, "function_call_config.yml")

    @property
    def function_output_serialization_path(self):
        return os.path.join(self.working_directory, "function_output.pkl")

    @property
    def job_state_path(self):
        return os.path.join(self.working_directory, "jobstate.yml")

    @property
    def function_call_output_lock_path(self):
        return os.path.join(self.working_directory, "function_call_output.lock")

    @property
    def function_has_returned_output(self):
        return self.bucket.exists(self.function_output_serialization_path)

    @property
    def job_has_failed(self):
        return ExceptionInJob.is_instance(self.job_output) or JobFailure.is_instance(
            self.job_output
        )

    def job_is_alive(self):
        data = self.bucket.get_blob(self.experiment_directory + "/heartbeat")
        if data is None:
            return False

        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()
            return True

        if data.updated > self.last_heartbeat + datetime.timedelta(seconds=30):
            return False

        self.last_heartbeat = data.updated
        return True

    def wait_till_output_is_ready(
        self, check_every: int = 5, timeout: Optional[int] = None
    ):
        start_time = time.time()
        while True:
            if self.job_is_alive:
                time.sleep(check_every)
                if timeout is not None and (time.time() - start_time) > timeout:
                    # We have timed out. Setting the flag below makes
                    # output_is_ready True, and as a consequence output is set to
                    # JobTimeout.
                    self.has_timed_out = True
            else:
                print("Job was not alive. Returning.")
                self.job_has_died = True
                return None
        return self.output

    wait = wait_till_output_is_ready

    def launch(self, tpu):
        self.tpu = tpu
        tpu.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.STARTING.value, "tpu_name": tpu.name}))

        # Run the job
        for cmd in self.tpu_job.setup:
            tpu.ssh(cmd, self.tpu_job.env)
        tpu.ssh(self.tpu_job.install, self.tpu_job.env)
        tpu.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.RUNNING.value, "tpu_name": tpu.name}), overwrite=True)

        train_cmd = f"{self.tpu_job.train_cmd} {self.bucket.name} {self.working_directory}"
        self.task = tpu.ssh(train_cmd, self.tpu_job.env, run_async=True)
        return self

    def clean_up(self):
        print(f"Cleaning up job: {self.working_directory}")
        print("Setting to FAILURE")
        self.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.FAILURE.value, "tpu_name": self.tpu.name}))
        return self

    def request_exit(self):
        if self.tpu_job is not None:
            self.tpu_job.request_exit()
        return self

    def __del__(self):
        self.clean_up()

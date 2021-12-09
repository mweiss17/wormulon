import io
import os
import uuid
import time
import asyncio
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
    job: TPUJob
    function_call: FunctionCall
    # Can be filled in later
    outbuffer = io.StringIO()
    errbuffer = io.StringIO()
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
        function_call = FunctionCall(fn, trainstate, job_kwargs)
        return cls(
            bucket=bucket,
            experiment_directory=exp_dir,
            function_call=function_call,
            tpu_job=tpu_job,
        )

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

    async def launch(self, tpu, check_every=1):
        self.tpu = tpu

        # Run the job
        for cmd in self.tpu_job.setup:
            tpu.ssh(cmd, self.tpu_job.env)
        tpu.ssh(self.tpu_job.install, self.tpu_job.env)
        tpu.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.RUNNING.value, "tpu_name": tpu.name}), overwrite=True)

        train_cmd = f"{self.tpu_job.train_cmd} {self.bucket.name} {self.working_directory}"
        proc = tpu.ssh(train_cmd, self.tpu_job.env, run_async=True)
        while True:
            out = proc.stdout.read()
            err = proc.stderr.read()
            self.outbuffer.write(out.decode("utf-8") if out else "")
            self.errbuffer.write(err.decode("utf-8") if err else "")
            if self.job_is_alive:
                await asyncio.sleep(check_every)
            else:
                print("Job was not alive. Returning.")
                self.job_has_died = True
                return None
        return self.output

    def clean_up(self):
        print(f"Cleaning up job: {self.working_directory}")
        for cmd in self.tpu_job.clean_up_cmds:
            self.tpu.ssh(cmd, self.tpu_job.env)
        self.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.FAILURE.value, "tpu_name": self.tpu.name}))
        return self



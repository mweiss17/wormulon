import os
import uuid
import time
from wormulon.utils import NotAvailable, ExceptionInJob, JobFailure, JobTimeout
from dataclasses import dataclass
from wormulon.fncall import FunctionCall
from wormulon.bucket import Bucket
from typing import Union, Any, Optional


@dataclass
class TPUJobHandler(object):
    # Required at instantiation
    job_id: str
    function_call: FunctionCall
    experiment_directory: str
    bucket: Bucket
    # Can be filled in later
    job_output: Union[Any, NotAvailable] = NotAvailable()
    has_timed_out: bool = False

    @classmethod
    def instantiate(
        cls, bucket: Bucket, experiment_directory: str, function_call: FunctionCall
    ):
        job_id = uuid.uuid4().hex
        return cls(
            job_id=job_id,
            bucket=bucket,
            experiment_directory=experiment_directory,
            function_call=function_call,
        )

    @property
    def working_directory(self):
        path = os.path.join(self.experiment_directory, self.job_id)
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
    def function_call_output_lock_path(self):
        return os.path.join(self.working_directory, "function_call_output.lock")

    @property
    def output_is_ready(self):
        return (
            os.path.exists(self.function_output_serialization_path)
            or self.has_timed_out
            or (self.tpu_job is not None and self.tpu_job.has_timed_out)
        )

    @property
    def function_has_returned_output(self):
        return os.path.exists(self.function_output_serialization_path)

    @property
    def output(self):
        if not isinstance(self.job_output, NotAvailable):
            # Reraise exception
            if ExceptionInJob.is_instance(self.job_output):
                raise Exception(self.job_output.exception)
            # Output has been read
            return self.job_output
        if not self.output_is_ready:
            # Output is not ready
            return
        # Output is ready but not read yet.
        # It can be a time-out, or a real output.
        if self.function_has_returned_output:
            self.job_output = self.function_call.deserialize_object(
                self.function_output_serialization_path
            )
        else:
            assert self.has_timed_out or (
                self.tpu_job is not None and self.tpu_job.has_timed_out
            ), "Expected a timeout, but didn't get one."
            self.job_output = JobTimeout()
        if ExceptionInJob.is_instance(self.job_output):
            raise Exception(self.job_output.exception)
        return self.job_output

    @property
    def job_has_failed(self):
        return ExceptionInJob.is_instance(self.job_output) or JobFailure.is_instance(
            self.job_output
        )

    def wait_till_output_is_ready(
        self, check_every: int = 5, timeout: Optional[int] = None
    ):
        start_time = time.time()
        while True:
            if self.output_is_ready:
                break
            else:
                time.sleep(check_every)
                if timeout is not None and (time.time() - start_time) > timeout:
                    # We have timed out. Setting the flag below makes
                    # output_is_ready True, and as a consequence output is set to
                    # JobTimeout.
                    self.has_timed_out = True
        return self.output

    wait = wait_till_output_is_ready

    def clean_up(self):
        shutil.rmtree(self.working_directory, ignore_errors=True)
        if self.tpu_job is not None:
            # nuke it for good measure
            self.tpu_job.nuke()
        return self

    def request_exit(self):
        if self.tpu_job is not None:
            self.tpu_job.request_exit()
        return self

    def __del__(self):
        self.clean_up()

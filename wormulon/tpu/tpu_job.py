import os
import time
import asyncio
import subprocess
from wormulon.core import Job
from wormulon.utils import execute, serialize, dump_yaml
from wormulon.utils import JobState
from wormulon.tpu.bucket import Bucket
from wormulon.tpu.fncall import FunctionCall

class TPUJob(Job):
    def __init__(
        self, trainer, train_state, tpu
    ):
        super().__init__()
        self.trainer = trainer
        self.bucket = Bucket(trainer.get("tpu/kwargs/bucket"))
        self.tpu = tpu
        try:
            self.train_state = self.bucket.get_latest_trainstate(trainer.experiment_directory)
        except IndexError:
            self.train_state = train_state
        self.function_call = FunctionCall(trainer, self.train_state, trainer.get("job/kwargs"))


    @property
    def working_directory(self):
        path = os.path.join(self.trainer.experiment_directory, self.job_id)
        return path

    @property
    def function_call_serialization_path(self):
        return os.path.join(self.working_directory, "function_call.pkl")

    @property
    def env(self):
        return self.trainer.get("job/kwargs/env_stmts")

    @property
    def install(self):
        return self.trainer.get("job/kwargs/install_cmd")

    @property
    def train(self):
        return self.trainer.get("job/kwargs/train_cmd")

    @property
    def setup(self):
        return self.trainer.get("job/kwargs/setup_cmds")

    @property
    def cleanup(self):
        return self.trainer.get("job/kwargs/cleanup_cmd")

    @property
    def job_state_path(self):
        return os.path.join(self.working_directory, "jobstate.yml")

    @property
    def failed(self):
        return ExceptionInJob.is_instance(self.job_output) or JobFailure.is_instance(
            self.job_output
        )

    @property
    def is_alive(self):
        data = self.bucket.get_blob(self.trainer.experiment_directory + "/heartbeat")

        # No heartbeat file means the job hasn't started yet
        if data is None:
            return True

        if data.updated > self.last_heartbeat + datetime.timedelta(seconds=30):
            return False

        self.last_heartbeat = data.updated
        return True

    def clean_up(self):
        print(f"Cleaning up job: {self.working_directory}")
        self.tpu.ssh(self.cleanup, self.env)
        self.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.FAILURE.value, "tpu_name": self.tpu.name}))
        return self

    def get_status(self):
        if self.last_heartbeat_at() > self.timeout:
            return JobState.FAILURE
        else:
            return JobState.SUCCESS
    #
    # def __del__(self):
    #     self.clean_up()

    async def launch(self, check_every=1):
        self.tpu.ssh(self.cleanup, self.env)

        self.tpu.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.STARTING.value, "tpu_name": self.tpu.name}))
        self.tpu.bucket.upload(self.function_call_serialization_path, self.function_call.serialize())

        # Run the job
        for cmd in self.setup:
            self.tpu.ssh(cmd, self.env)
        self.tpu.ssh(self.install, self.env)
        self.tpu.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.RUNNING.value, "tpu_name": self.tpu.name}), overwrite=True)

        train_cmd = f"{self.train} {self.bucket.name} {self.working_directory}"
        proc = self.tpu.ssh(train_cmd, self.env, run_async=True)

        while True:
            out = proc.stdout.read()
            err = proc.stderr.read()
            self.outbuffer.write(out.decode("utf-8") if out else "")
            self.errbuffer.write(err.decode("utf-8") if err else "")
            if self.is_alive:
                await asyncio.sleep(check_every)
            else:
                print("Job was not alive. Returning.")
                self.died = True
                return None
        return self.output

    def submit(self):
        return asyncio.ensure_future(self.launch())

import os
import time
import asyncio
import subprocess
import datetime
from wormulon.core import Job
from wormulon.utils import execute, serialize, dump_yaml
from wormulon.utils import JobState
from wormulon.tpu.bucket import Bucket
from wormulon.tpu.fncall import FunctionCall

class TPUJob(Job):
    def __init__(
        self, trainer, tpu
    ):
        super().__init__()
        self.trainer = trainer
        self.bucket = Bucket(trainer.get("tpu/kwargs/bucket"))
        self.tpu = tpu
        self.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.STARTING.value, "tpu_name": self.tpu.name}), overwrite=True)
        self.train_state = None

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
        return ExceptionInJob.is_instance(self.function_call.outputs) or JobFailure.is_instance(
            self.function_call.outputs
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
        self.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.FAILURE.value, "tpu_name": self.tpu.name}), overwrite=True)
        return self

    def get_status(self):
        if self.last_heartbeat_at() > self.timeout:
            return JobState.FAILURE
        else:
            return JobState.SUCCESS
    #
    # def __del__(self):
    #     self.clean_up()

    async def nonblocking_ssh(self, cmd, env, check_every=1):
        proc = self.tpu.ssh(cmd, env, run_async=True)

        while True:
            out = proc.stdout.read()
            err = proc.stderr.read()
            out = out.decode("utf-8") if out else ""
            err = err.decode("utf-8") if err else ""
            self.outbuffer.write(out)
            self.errbuffer.write(err)
            poll = proc.poll()
            if poll is None:
                await asyncio.sleep(check_every)
            elif "Finished worker" in self.outbuffer.getvalue():
                return True
            else:
                return poll

    def arm(self, train_state, resume=False):
        print("Arming job")
        self.train_state = train_state

        if resume:
            try:
                self.train_state = self.bucket.get_latest_trainstate(self.trainer.experiment_directory)
            except IndexError:
                pass
        if not self.tpu.is_ready:
            self.tpu.create()

    async def launch(self, check_every=1):
        await self.nonblocking_ssh(self.cleanup, self.env)
        assert self.train_state is not None
        self.function_call = FunctionCall(self.trainer, self.train_state, self.trainer.get("job/kwargs"), self.tpu.name)
        self.tpu.bucket.upload(self.function_call_serialization_path, self.function_call.serialize())

        # setup the TPU
        for cmd in self.setup:
            print(f"Running setup command: {cmd}")
            stdout, stderr, retcode = self.tpu.ssh(cmd, self.env, capture_output=True)
            # If we need to install everything we get an error and then do it here
            if retcode == 1:
                print(f"Installing {self.install}")
                await self.nonblocking_ssh(self.install, self.env)
                break
        self.tpu.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.RUNNING.value, "tpu_name": self.tpu.name}), overwrite=True)

        train_cmd = f"{self.train} {self.bucket.name} {self.working_directory}"
        print(f"Running train command: {train_cmd}")
        await self.nonblocking_ssh(train_cmd, self.env, check_every=10)
        return self.function_call.outputs

    def submit(self):
        return asyncio.ensure_future(self.launch())

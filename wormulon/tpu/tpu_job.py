import os
import time
import subprocess
from wormulon.core import Job
from wormulon.utils import execute, serialize
from wormulon.utils import JobState


class TPUJob(Job):
    def __init__(
        self, setup_cmds, install_cmd, env_stmts, cleanup_cmds, **kwargs
    ):
        super().__init__(**kwargs)
        self.setup_cmds = setup_cmds
        self.install_cmd = install_cmd
        self.env_stmts = env_stmts
        self.cleanup_cmds = cleanup_cmds

    @property
    def env(self):
        return self.env_stmts

    @property
    def install(self):
        return self.install_cmd

    @property
    def train(self):
        return self.train_cmd

    @property
    def setup(self):
        return self.setup_cmds

    @property
    def cleanup(self):
        return self.cleanup_cmds

    def last_heartbeat_at(self, relative_to_now=False):
        if relative_to_now:
            return time.time() - self.last_heartbeat
        else:
            return self.last_heartbeat

    def get_status(self):
        if self.last_heartbeat_at() > self.timeout:
            return JobState.FAILURE
        else:
            return JobState.SUCCESS

    def read_output(self):
        if not os.path.exists(self.out_path):
            return None
        with open(self.out_path, "r") as f:
            output = f.read()
        return output

    def has_timed_out(self):
        return self.last_heartbeat_at() > self.timeout

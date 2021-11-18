import time
import subprocess

try:
    import torch_xla.core.xla_model as xm
except Exception:
    xm = None

from wormulon.utils import JobStatus


class Job:
    def __init__(self, job_id, timeout=60):
        self.job_id = job_id
        self.timeout = timeout

    def __repr__(self):
        return "<Job: {}>".format(self.job_id)

    def __str__(self):
        return "<Job: {}>".format(self.job_id)

    def nuke(self):
        pass

    def check(self):
        self.run()

    def last_heartbeat_at(self, relative_to_now=False):
        if relative_to_now:
            return time.time() - self.last_heartbeat
        else:
            return self.last_heartbeat

    def get_status(self):
        if self.last_heartbeat_at() > self.timeout:
            return JobStatus.FAILURE
        else:
            return JobStatus.SUCCESS


class SlurmJob(Job):
    def __init__(self, job_id, command, dry=True):
        super().__init__(job_id, command, dry)

    def nuke(self):
        print(f"Nuking job {self.job_id}")
        bash_command = "scancel {self.job_id}"
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


class Node(object):
    def __init__(self, name, jobs=None):
        self.name = name
        self.jobs = jobs or []

    def __repr__(self):
        return "<Node: {}>".format(self.name)

    def delete(self):
        pass

    def create(self):
        pass

    def ssh(self):
        pass

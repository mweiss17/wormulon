import io
import uuid
import time
import subprocess
from wormulon.utils import JobState


class Job:
    def __init__(self, timeout=60):
        self.job_id = uuid.uuid4().hex
        self.last_heartbeat = time.time()
        self.timeout = timeout
        self.outbuffer = io.StringIO()
        self.errbuffer = io.StringIO()

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
            return JobState.FAILURE
        else:
            return JobState.SUCCESS


    @property
    def has_timed_out(self):
        return self.last_heartbeat_at > self.timeout


class SlurmJob(Job):
    def __init__(self, job_id, command, dry=True):
        super().__init__(job_id, command, dry)

    def nuke(self):
        print(f"Nuking job {self.job_id}")
        bash_command = "scancel {self.job_id}"
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output = process.communicate()


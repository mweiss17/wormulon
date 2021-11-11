import time
import subprocess
from wormulon.utils import JobStatus


class Job:
    def __init__(self, job_id, command, dry=True):
        self.job_id = job_id
        self.command = command
        self.dry = dry
        self.last_heartbeat = time.time()

    def __repr__(self):
        return "<Job: {}>".format(self.job_id)

    def __str__(self):
        return "<Job: {}>".format(self.job_id)

    def nuke(self):
        pass

    def run(self):
        print("Running job {}".format(self.job_id))
        self.command()

    def check(self):
        if self.dry:
            print("Dry run for job {}".format(self.job_id))
            return

        self.run()

    def last_heartbeat_at(self, relative_to_now=False):
        if relative_to_now:
            return time.time() - self.last_heartbeat
        else:
            return self.last_heartbeat

    def get_status(self):
        if self.last_heartbeat_at() > 30:
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


class TPUJob(Job):
    def __init__(self, job_id, command, zone, dry=True):
        super().__init__(job_id, command, dry)
        self.zone = zone

    def nuke(self):
        print(f"Nuking job {self.job_id}")
        bash_command = (
            f"gcloud alpha compute tpus tpu-vm delete {self.job_id} --zone={self.zone}"
        )
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output, error

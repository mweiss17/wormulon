import os
import asyncio
from wormulon.utils import execute, JobState, serialize, dump_yaml
from wormulon.tpu.bucket import Bucket
from wormulon.tpu.tpu import TPU
from wormulon.tpu.tpu_job import TPUJob
from wormulon.tpu.fncall import FunctionCall


class TPUManager(object):
    def __init__(self, **kwargs):
        self.bucket = Bucket(kwargs.get("bucket"))
        self.zone = kwargs.get("zone")
        self.project = kwargs.get("project")
        self.tpu_kwargs = kwargs

    @property
    def tpu_ids(self):
        command = f"gcloud alpha compute tpus list --zone={self.zone} --format=value[seperator=','](name)"
        stdout, stderr, retcode = execute(command.split(), capture_output=True)
        ids = stdout.split("\n")
        ids.remove("")
        int_ids = [-1]
        int_ids.extend([int(i.split("-")[-1]) for i in ids])
        return int_ids

    @property
    def ready_tpus(self):
        command = f"gcloud compute tpus list --format=value(NAME,STATUS) --zone {self.zone}"
        stdout, stderr, retcode = execute(command.split(), capture_output=True)
        rows = stdout.split("\n")
        rows.remove("")
        names = {r.split("\t")[0] for r in rows if r.split("\t")[1] == "READY"}
        return names

    @property
    def busy_tpus(self):
        return {job.get("tpu_name") for job in self.bucket.list_jobs(filters=[JobState.RUNNING, JobState.STARTING])}

    @property
    def available_tpus(self):
        return self.ready_tpus - self.busy_tpus

    def get_or_create_tpu(self):
        """ Returns a TPU object if one is available (not running a job), otherwise creates a new one """
        try:
             name = self.available_tpus.pop()
             print(f"Using existing TPU {name}")
             tpu = TPU(name, **self.tpu_kwargs)
        except Exception:
            name = f"{self.project}-{max(self.tpu_ids) + 1}"
            print(f"Creating new tpu {name}")
            tpu = TPU(name, **self.tpu_kwargs)
            tpu.create()
        return tpu


    def get_tpus(self, num_tpus):
        tpus = []
        available_tpus = self.available_tpus.copy()

        for _ in range(num_tpus):
            if any(available_tpus):
                name = available_tpus.pop()
                print(f"Using existing TPU {name}")
                tpu = TPU(name, **self.tpu_kwargs, is_ready=True)
            else:
                name = f"{self.project}-{max(self.tpu_ids) + 1}"
                print(f"Creating new tpu {name}")
                tpu = TPU(name, **self.tpu_kwargs, is_ready=False)
            tpus.append(tpu)

        return tpus

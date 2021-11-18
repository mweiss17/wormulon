from collections import defaultdict
from wormulon.utils import execute
from wormulon.bucket import Bucket
from wormulon.tpu import TPU


class TPUZoneManager(object):
    def __init__(self, **kwargs):
        self.bucket = Bucket(kwargs.get("bucket"))
        self.zone = kwargs.get("zone")
        self.tpu_kwargs = kwargs
        self.tpus = self.get_all_tpus()
        self._jobs = None

    def get_available_tpu(self):
        unavailable_names = set()
        for job in self.jobs["running"]:
            unavailable_names.add(job.config.get("tpu_name"))

        # Find an available tpu
        for tpu in self.tpus:
            if tpu.name not in unavailable_names:
                return tpu

        # otherwise return a new TPU
        return TPU(**self.tpu_kwargs)

    def get_all_tpus(self):
        command = f"gcloud compute tpus list --format=value(name)"
        output, error = execute(command.split())
        names = output.decode("utf-8").strip().split("\n")
        tpus = []
        for name in names:
            if name == "":
                continue
            tpus.append(TPU(name=name, **self.tpu_kwargs))
        return tpus

    @property
    def jobs(self):
        if self._jobs is not None:
            return self._jobs

        jobs = defaultdict(list)
        runs = self.bucket.list(filter="")
        for run in runs:
            jobs[run.state].append(run)
        self._jobs = jobs
        s = ""
        for k, v in jobs.items():
            s += f"{k}: {len(v)}, "
        print(s)
        return self._jobs

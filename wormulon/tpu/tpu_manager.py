from wormulon.utils import execute, JobState
from wormulon.tpu.bucket import Bucket
from wormulon.tpu.tpu import TPU


class TPUManager(object):
    def __init__(self, **kwargs):
        self.bucket = Bucket(kwargs.get("bucket"))
        self.zone = kwargs.get("zone")
        self.project = kwargs.get("project")
        self.tpu_kwargs = kwargs
        self.disarmed_but_busy_tpus = set()

    @property
    def tpu_ids(self):
        command = f"gcloud alpha compute tpus list --zone={self.zone} --format=value[seperator=','](name)"
        stdout, stderr, retcode = execute(command.split(), capture_output=True)
        ids = stdout.split("\n")
        ids.remove("")
        int_ids = [-1]
        int_ids.extend([int(i.split("-")[-1]) for i in ids])
        int_ids.extend([int(i.split("-")[-1]) for i in self.disarmed_but_busy_tpus])
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
        return {job.get("tpu_name") for job in self.bucket.list_jobs(filters=[JobState.RUNNING, JobState.STARTING, JobState.ARMED])}

    @property
    def available_tpus(self):
        return (self.ready_tpus - self.busy_tpus) - self.disarmed_but_busy_tpus

    def get_tpus(self, num_tpus):
        tpus = []
        available_tpus = self.available_tpus.copy()
        # print(f"available_tpus: {available_tpus}, disarmed_but_busy: {self.disarmed_but_busy_tpus}")
        for _ in range(num_tpus):
            if any(available_tpus):
                name = available_tpus.pop()
                print(f"Using existing TPU {name}")
                tpu = TPU(name, **self.tpu_kwargs)
            else:
                name = f"{self.project}-{max(self.tpu_ids) + 1}"
                print(f"Creating new tpu {name}")
                tpu = TPU(name, **self.tpu_kwargs)
                self.disarmed_but_busy_tpus.add(tpu.name)
            tpus.append(tpu)
        return tpus

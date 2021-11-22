from wormulon.utils import execute, JobState, serialize
from wormulon.tpu.bucket import Bucket
from wormulon.tpu.tpu import TPU
from wormulon.tpu.tpu_job import TPUJob
from wormulon.tpu.tpu_handler import TPUJobHandler
from wormulon.tpu.fncall import FunctionCall


class TPUManager(object):
    def __init__(self, **kwargs):
        self.bucket = Bucket(kwargs.get("bucket"))
        self.zone = kwargs.get("zone")
        self.project = kwargs.get("project")
        self.tpu_kwargs = kwargs
        self.tpus = self.get_all_tpus()

    def get_available_tpu(self):
        unavailable_names = set()

        for job in self.bucket.list_jobs(filter=JobState.RUNNING):
            unavailable_names.add(job.config.get("tpu_name"))

        # Find an available tpu
        for tpu in self.tpus:
            if tpu.name not in unavailable_names:
                return tpu

        # otherwise create a new TPU
        ids, error = self.get_tpu_ids()
        name = f"{self.project}-{max(ids) + 1}"
        new_tpu = TPU(name, **self.tpu_kwargs)
        new_tpu.create()
        return new_tpu

    def get_all_tpus(self):
        command = f"gcloud compute tpus list --format=value(name) --zone {self.zone}"
        stdout, stderr, retcode = execute(command.split())
        names = stdout.split("\n")
        tpus = []
        for name in names:
            if name == "":
                continue
            tpus.append(TPU(name=name, **self.tpu_kwargs))
        return tpus

    def get_tpu_ids(self):
        command = f"gcloud alpha compute tpus list --zone={self.zone} --format=value[seperator=','](name)"
        stdout, stderr, retcode = execute(command.split())
        ids = stdout.split("\n")
        ids.remove("")
        int_ids = [-1]
        int_ids.extend([int(i.split("-")[-1]) for i in ids])
        return int_ids

    def submit(self, trainer, training_state, **job_kwargs):

        # Get a TPU
        if job_kwargs.get("tpu_name") is not None:
            tpu = TPU(job_kwargs.get("tpu_name"), **self.tpu_kwargs)
        else:
            tpu = self.get_available_tpu()

        # Create a handler
        handler = TPUJobHandler.instantiate(
            function_call=FunctionCall(trainer, training_state, job_kwargs),
        )
        self._job_handlers.append(handler)
        tpu.bucket.upload(
            handler.function_call_serialization_path, handler.function_call
        )

        # Create a job
        job = TPUJob(**self.get("job/kwargs"))

        # Run the job
        for cmd in job.setup:
            tpu.ssh(cmd, job.env)
        tpu.ssh(job.install, job.env)

        tpu.ssh(
            f"{job.train} {self.bucket.name} {handler.function_call_serialization_path}"
        )
        return handler

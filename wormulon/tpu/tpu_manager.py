import os
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
        self.tpus = self.get_all_ready_tpus()
        self._job_handlers = []

    def get_available_tpu(self):
        unavailable_names = set()
        for job in self.bucket.list_jobs(filter=JobState.RUNNING):
            unavailable_names.add(job.get("tpu_name"))

        # Find an available tpu
        for tpu in self.tpus:
            if tpu.name not in unavailable_names:
                return tpu

        # otherwise create a new TPU
        ids = self.get_tpu_ids()
        name = f"{self.project}-{max(ids) + 1}"
        print(f"No available tpus, creating {name}")
        new_tpu = TPU(name, **self.tpu_kwargs)
        new_tpu.create()
        return new_tpu

    def get_all_ready_tpus(self):
        command = f"gcloud compute tpus list --format=value(NAME,STATUS) --zone {self.zone}"
        stdout, stderr, retcode = execute(command.split(), capture_output=True)
        rows = stdout.split("\n")
        rows.remove("")
        ready_names = [r.split("\t")[0] for r in rows if r.split("\t")[1] == "READY"]
        tpus = []
        for name in ready_names:
            tpus.append(TPU(name=name, **self.tpu_kwargs))
        return tpus

    def get_tpu_ids(self):
        command = f"gcloud alpha compute tpus list --zone={self.zone} --format=value[seperator=','](name)"
        stdout, stderr, retcode = execute(command.split(), capture_output=True)
        ids = stdout.split("\n")
        ids.remove("")
        int_ids = [-1]
        int_ids.extend([int(i.split("-")[-1]) for i in ids])
        return int_ids

    def submit(self, fn, trainstate, exp_dir, **job_kwargs):

        # Get a TPU
        existing_tpu_name = job_kwargs.get("tpu_name")
        if existing_tpu_name is not None:
            tpu = TPU(existing_tpu_name, **self.tpu_kwargs)
            if tpu.name not in [t.name for t in self.get_all_ready_tpus()]:
                tpu.create()
        else:
            tpu = self.get_available_tpu()
        print(f"running on {tpu.name}")

        # Try to Add WANDB_API_KEY
        job_kwargs['env_stmts'].append(
            f"export WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')};"
        )


        # Check if this experiment has a checkpoint
        # experiments = self.bucket.list_experiments()
        # found = False
        # exp_id = f'{exp_dir.split("/")[-1]}-{fn.get("dataset/kwargs/name")}'
        # for eid, exp in experiments.items():
        #     if exp_id == eid:
        #         print(f"Resuming from {exp}")
        #         found = True
        #         break
        # if found:
        #     function_call = FunctionCall(fn, exp.blob.name, job_kwargs)
        # else:
        # function_call = FunctionCall(fn, trainstate, job_kwargs)
        handler = TPUJobHandler.instantiate(self.bucket, exp_dir, fn, trainstate, job_kwargs)
        tpu.bucket.upload(handler.function_call_serialization_path, handler.function_call.serialize())
        self._job_handlers.append(handler)
        return handler.launch(tpu)

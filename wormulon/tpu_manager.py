from wormulon.utils import execute, JobState, serialize
from wormulon.bucket import Bucket
from wormulon.tpu import TPU


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

    def launch(self, job, tpu_name=None):
        if tpu_name is not None:
            tpu = TPU(tpu_name, **self.tpu_kwargs)
        else:
            tpu = self.get_available_tpu()

        # update the configuration on wandb noting this tpu's name
        # job.trainer._config["tpu_name"] = tpu.name
        # job.trainer.update_wandb_config()

        self.bucket.upload(job.path, serialize(job))

        # upload the job to GCP storage
        for cmd in job.setup_cmds:
            tpu.ssh(cmd, job.env_stmts)
        tpu.ssh(job.install_cmd, job.env_stmts)

        tpu.ssh(f"{job.train_cmd} {self.bucket.name} {job.path}")


class TPUJob(object):
    def __init__(
        self,
        path,
        trainer,
        trainstate,
        setup_cmds,
        install_cmd,
        train_cmd,
        env_stmts,
        cleanup_cmds,
    ):
        self.path = path
        self.trainer = trainer
        self.trainstate = trainstate
        self.setup_cmds = setup_cmds
        self.install_cmd = install_cmd
        self.train_cmd = train_cmd
        self.env_stmts = env_stmts
        self.cleanup_cmds = cleanup_cmds

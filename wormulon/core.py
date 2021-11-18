import io
import torch
import time
import wandb
import subprocess
from collections import defaultdict

try:
    import torch_xla.core.xla_model as xm
except Exception:
    xm = None

from wormulon.utils import (
    JobStatus,
    execute,
    get_tpu_ids,
    _upload_data_to_gcs,
    _read_blob_gcs,
    _check_exists_gcs,
    _delete_blob_gcs,
)


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


class TPU(Node):
    def __init__(
        self, zone, network, subnet, netrange, acc_type, preemptible, name=None
    ):
        self.zone = zone
        self.network = network
        self.subnet = subnet
        self.netrange = netrange
        self.acc_type = acc_type
        self.preemptible = preemptible
        self.history = []
        self._wandb_api_key = None

        if name is not None:
            self.name = name
        else:
            node_ids, error = get_tpu_ids()
            self.name = f"node-{max(node_ids) + 1}"
            self.create()

    def delete(self):
        return
        return execute(
            f"gcloud alpha compute tpus tpu-vm delete {self.name} --zone {self.zone}".split()
        )

    def create(self, retry=True):
        while True:
            command = f"gcloud alpha compute tpus tpu-vm create {self.name} \
              --zone {self.zone} \
              --network {self.network} \
              --subnetwork {self.subnet} \
              --range {self.netrange} \
              --accelerator-type {self.acc_type} \
              --version v2-alpha"

            if self.preemptible:
                command += " --preemptible"

            output, error = execute(command.split())
            print(f"error was: {error}")
            if error is not None:
                breakpoint()
                print(f"Error creating TPU: {error}")
                if retry:
                    time.sleep(5)
                    continue
                else:
                    return output, error
            else:
                return output, error
        return output, error

    @property
    def wandb_api_key(self):
        if not self._wandb_api_key:
            self._wandb_api_key, err = self.ssh(
                "curl http://metadata.google.internal/computeMetadata/v1/project/attributes/wandb_api_key -H Metadata-Flavor:Google",
                use_env=False,
            )
            self._wandb_api_key = self._wandb_api_key.decode("utf-8").strip()
        return self._wandb_api_key

    @property
    def env(self):
        env = 'export XRT_TPU_CONFIG="localservice;0;localhost:51011";'
        env += "export PATH=$PATH:/home/$USER/.local/bin;"
        env += f"export WANDB_API_KEY={self.wandb_api_key};"
        env += "unset LD_PRELOAD;"

        return env

    def ssh(self, cmd, use_env=True):
        command = (
            f"gcloud alpha compute tpus tpu-vm ssh "
            f"{self.name} "
            f"--zone {self.zone} "
            f"--command "
        )
        command = command.split()
        if use_env:
            cmd = self.env + cmd
        command.append(cmd)
        print(f"running: {command} on {self.name}")

        output, error = execute(command)
        return output, error

    @property
    def internal_ip(self):
        command = f"gcloud compute tpus describe {self.name} --zone {self.zone} --format=value(networkInterfaces[0].networkIP)"
        output, error = execute(command.split())
        return output.decode("utf-8").strip()


class TPUJob(Job):
    def __init__(
        self,
        job_id,
        experiment_directory,
        bucket,
        trainer,
        training_state,
        timeout=3600,
    ):
        super().__init__(job_id)
        self.experiment_directory = experiment_directory
        self.bucket = GCSBucket(bucket)
        self.trainer = trainer
        self.training_state = training_state
        self.timeout = timeout

    def __repr__(self):
        return "<TPUJob: {}>".format(self.job_id)

    @property
    def prefix(self):
        self._prefix = f"{self.experiment_directory}/e{self.training_state.epoch}_s{self.training_state.step}"
        return self._prefix

    @property
    def path(self):
        return f"{self.prefix}_{self.trainer.get('dataset/kwargs/name')}.pt"

    def serialize(self):
        buffer = io.BytesIO()
        if xm:
            xm.save(self, buffer)
        else:
            torch.save(self, buffer)
        return buffer.getvalue()

    def beat(self):
        data = torch.dumps({"last_heartbeat": time.time()})
        self.bucket.upload(self.prefix + "heartbeat.pt", data)

    @property
    def last_heartbeat(self):
        data = torch.load(self.bucket.download(self.prefix + "heartbeat.pt"))
        return data["last_heartbeat"]

    @property
    def status(self):
        if self.last_heartbeat + self.timeout < time.time():
            return JobStatus.TIMEOUT
        elif self.last_heartbeat + self.timeout > time.time() + 60:
            return JobStatus.RUNNING
        elif self.last_heartbeat + self.timeout > time.time() + 30:
            return JobStatus.STARTING
        elif self.preempted:
            return JobStatus.PREEMPTED
        else:
            return JobStatus.FAILURE

    @property
    def preempted(self):
        # TODO: check if preempted
        # command = f"gcloud compute operations list --filter=operationType=compute.instances.preempted"
        command = f"gcloud compute tpus describe {self.name} --format=value(status)"
        output, error = execute(command.split())
        print(output)
        if output == "PREEMPTED":
            return True
        else:
            return False

    def wait(self):
        while True:
            # if self.preempted:
            #     print("Preempted")
            #     return JobStatus.PREEMPTED
            # if self.done:
            #     print("Done")
            #     return JobStatus.DONE
            # if time.time() - self.last_heartbeat > self.timeout:
            #     print("No heartbeat")
            #     return JobStatus.FAILED
            time.sleep(10)

    def clean_up(self):
        pass
        # self.bucket.delete()

    def upload(self, overwrite=False):
        buffer = self.serialize()
        if len(buffer) > 0:
            self.bucket.upload(self.path, buffer, overwrite=overwrite)


class GCSBucket(object):
    def __init__(self, name):
        self.name = name

    def upload(self, path, data, overwrite=False):
        if self.exists(path) and not overwrite:
            print(f"{path} already exists")
            return
        _upload_data_to_gcs(self.name, path, data)
        print(f"Uploading {self.name}/{path}")

    def download(self, path):
        return _read_blob_gcs(self.name, path)

    def exists(self, path):
        return _check_exists_gcs(self.name, path)

    def delete(self, path):
        _delete_blob_gcs(self.name, path)


class TPUCluster(object):
    def __init__(self, wandb_entity, wandb_project, **kwargs):
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.bucket = kwargs.get("bucket")
        self.zone = kwargs.get("zone")
        self.tpu_kwargs = kwargs
        self.tpus = self.get_all_tpus()
        self._jobs = None

    def get_available_tpu(self, run_id):
        unavailable_names = set()
        for job in self.jobs["running"]:
            if run_id == job._attrs.get("name"):
                continue  # skip this job
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

        api = wandb.Api()
        runs = api.runs(self.wandb_entity + "/" + self.wandb_project)
        jobs = defaultdict(list)

        for run in runs:
            jobs[run.state].append(run)
        self._jobs = jobs
        s = ""
        for k, v in jobs.items():
            s += f"{k}: {len(v)}, "
        print(s)
        return self._jobs

    def run(self, job: TPUJob):
        tpu = self.get_inactive_tpus()[0]
        tpu.run(job)

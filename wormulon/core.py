import io
import torch
import time
import subprocess
from wormulon.utils import (
    JobStatus,
    execute,
    get_tpu_ids,
    _upload_data_to_gcs,
    _read_blob_gcs,
    _check_exists_gcs,
)


class Job:
    def __init__(self, job_id, command, timeout=60):
        self.job_id = job_id
        self.command = command
        self.last_heartbeat = time.time()
        self.timeout = timeout

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
        self, bucket, zone, network, subnet, netrange, acc_type, preemptible, asynch,
    ):
        self.bucket = GCSBucket(bucket)
        self.zone = zone
        self.network = network
        self.subnet = subnet
        self.netrange = netrange
        self.acc_type = acc_type
        self.preemptible = preemptible
        self.asynch = asynch
        self.history = []
        node_ids, error = get_tpu_ids()
        self.name = f"node-{max(node_ids) + 1}"

    def delete(self):
        return
        return execute(
            f"gcloud alpha compute tpus tpu-vm delete {self.name} --zone {self.zone}".split()
        )

    def create(self, retry=True):
        # If it's asynchronous, we can't retry (otherwise Google will be sad)
        while True and not self.asynch:
            command = f"gcloud alpha compute tpus tpu-vm create {self.name} \
              --zone {self.zone} \
              --network {self.network} \
              --subnetwork {self.subnet} \
              --range {self.netrange} \
              --accelerator-type {self.acc_type} \
              --version v2-alpha"

            if self.preemptible:
                command += " --preemptible"
            if self.asynch:
                command += " --async"

            output, error = execute(command.split())
            if error:
                print(f"Error creating TPU: {error}")
                if retry:
                    time.sleep(5)
                    continue
                else:
                    return output, error
            else:
                return output, error
        return output, error

    def ssh(self, cmd):
        command = (
            f"gcloud alpha compute tpus tpu-vm ssh "
            f"{self.name} "
            f"--zone {self.zone} "
            f"--command "
        )

        command = command.split()
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
        wandb_run_id,
        experiment_directory,
        tpu,
        trainer,
        training_state,
        install_cmd,
        train_cmd,
    ):
        super().__init__(wandb_run_id, train_cmd)
        self.experiment_directory = experiment_directory
        self.tpu = tpu
        self.trainer = trainer
        self.training_state = training_state
        self.install_cmd = install_cmd
        self.train_cmd = train_cmd
        self.last_heartbeat = time.time()

    def __repr__(self):
        return "<TPUJob: {}>".format(self.job_id)

    def create(self):
        self.tpu.create()

    def ssh(self, cmd):
        self.tpu.ssh(cmd)

    def install(self):
        self.ssh(self.install_cmd)

    def train(self):
        train_args = " ".join([self.tpu.bucket, self.tpu_job_path])
        self.ssh(self.train_cmd + " " + train_args)

    @property
    def prefix(self):
        self._prefix = f"{self.experiment_directory}/e{self.training_state.epoch}_s{self.training_state.step}"
        return self._prefix

    @property
    def tpu_job_path(self):
        return f"{self.prefix}/{self.job_id}.pt"

    def upload(self, overwrite=False):
        if self.tpu.bucket.exists(self.tpu_job_path) and not overwrite:
            print(f"{self.tpu_job_path} already exists")
            return
        self.tpu.bucket.upload(self.tpu_job_path, self.serialize())

    def serialize(self):
        buffer = io.BytesIO()
        torch.save(self, buffer)
        return buffer.getvalue()

    @property
    def preempted(self):
        # TODO: check if preempted
        # command = f"gcloud compute operations list --filter=operationType=compute.instances.preempted"
        command = f"gcloud compute tpus describe {self.tpu.name} --format=value(status)"
        output, error = execute(command.split())
        print(output)
        if output == "PREEMPTED":
            return True
        else:
            return False

    @property
    def done(self):
        # TODO implement (check if number of steps is reached)
        pass

    @property
    def failed(self):
        # TODO implement by checking the heartbeat on the tpu
        pass

    def wait(self):
        while True:
            if self.preempted:
                print("Preempted")
                return JobStatus.PREEMPTED
            if self.done:
                print("Done")
                return JobStatus.DONE
            if time.time() - self.last_heartbeat > self.timeout:
                print("No heartbeat")
                return JobStatus.FAILED
            time.sleep(10)

    def clean_up(self):
        self.tpu.delete()


class GCSBucket(object):
    def __init__(self, bucket):
        self.bucket = bucket

    def upload(self, path, data):
        _upload_data_to_gcs(self.bucket, path, data)
        print(f"Uploading {self.bucket}/{path}")

    def download(self, path):
        return _read_blob_gcs(self.bucket, path)

    def exists(self, path):
        return _check_exists_gcs(self.bucket, path)

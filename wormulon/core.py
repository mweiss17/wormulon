import time
import subprocess
from wormulon.utils import JobStatus, execute, get_tpu_ids, _upload_data_to_gcs


class Job:
    def __init__(self, job_id, command):
        self.job_id = job_id
        self.command = command
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
        self.bucket = bucket
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
        print(f"deleting: {self.name}")
        command = (
            f"gcloud compute tpus tpu-vm delete {self.name} --zone {self.zone} --async"
        )
        output, error = execute(command.split())
        return output, error

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
        job_id,
        experiment_directory,
        tpu,
        trainer,
        training_state,
        install_cmd,
        train_cmd,
    ):
        super().__init__(job_id, train_cmd)
        self.experiment_directory = experiment_directory
        self.tpu = tpu
        self.trainer = trainer
        self.training_state = training_state
        self.install_cmd = install_cmd
        self.train_cmd = train_cmd
        self.last_heartbeat = time.time()
        self._trainer_path = None
        self._training_state_path = None
        self.train_args = " ".join(
            [self.tpu.bucket, self.trainer_path, self.training_state_path]
        )

    def __repr__(self):
        return "<TPUJob: {}>".format(self.job_id)

    def create(self):
        self.tpu.create()

    def ssh(self, cmd):
        self.tpu.ssh(cmd)

    def install(self):
        self.ssh(self.install_cmd)

    def train(self):
        self.ssh(self.train_cmd + " " + self.train_args)

    @property
    def trainer_path(self):
        if self._trainer_path is not None:
            return self._trainer_path
        prefix = f"{self.experiment_directory}/{str(time.time())}"
        trainer_path = f"{prefix}/trainer.pt"
        return trainer_path

    @property
    def training_state_path(self):
        if self._training_state_path is not None:
            return self._training_state_path
        prefix = f"{self.experiment_directory}/{str(time.time())}"
        training_state_path = f"{prefix}/training_state.pt"
        return training_state_path

    def upload(self):
        print(f"Uploading {self.tpu.bucket}/{self.training_state_path}")
        _upload_data_to_gcs(
            "gs://" + self.tpu.bucket, self.trainer_path, self.trainer.serialize()
        )
        _upload_data_to_gcs(
            "gs://" + self.tpu.bucket,
            self.training_state_path,
            self.training_state.serialize(),
        )

    @property
    def preempted(self):
        # command = f"gcloud compute operations list --filter=operationType=compute.instances.preempted"
        command = f"gcloud compute tpus describe {self.tpu.name} --format=value(status)"
        output, error = execute(command.split())
        print(output)
        if output == "PREEMPTED":
            return True
        else:
            return False

    def wait(self):
        while True:
            if self.preempted:
                print("Preempted")
                return
            if self.done:
                print("Done")
                return
            if time.time() - self.last_heartbeat > 60:
                print("No heartbeat")
                return
            time.sleep(1)

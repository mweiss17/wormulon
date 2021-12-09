import torch
import time
import asyncio
from wormulon.tpu.fncall import FunctionCall
from wormulon.tpu.tpu_handler import TPUJobHandler
from wormulon.utils import execute, serialize
from wormulon.tpu.bucket import Bucket

class TPU:
    def __init__(
        self,
        name,
        zone,
        network,
        subnet,
        netrange,
        acc_type,
        preemptible,
        bucket,
        project,
    ):
        self.name = name
        self.zone = zone
        self.network = network
        self.subnet = subnet
        self.netrange = netrange
        self.acc_type = acc_type
        self.preemptible = preemptible
        self.bucket = Bucket(bucket)


    def __repr__(self):
        return f"TPU({self.name})"

    def delete(self):
        print(f"deleting tpu {self.name}")
        return execute(
            f"gcloud alpha compute tpus tpu-vm delete {self.name} --zone {self.zone} --async".split()
        )

    def create(self, retry=True):
        while True:
            command = f"gcloud alpha compute tpus tpu-vm create {self.name} \
              --zone {self.zone} \
              --network {self.network} \
              --subnetwork {self.subnet} \
              --range {self.netrange} \
              --accelerator-type {self.acc_type} \
              --version tpu-vm-pt-1.10"

            if self.preemptible:
                command += " --preemptible"

            stdout, stderr, retcode = execute(command.split(), capture_output=True)
            if retcode == 0:
                return stdout
            else:
                if retry:
                    time.sleep(10)
                else:
                    return stderr

    def ssh(self, cmd, env_stmts=[], run_async=False, capture_output=False):
        command = (
            f"gcloud alpha compute tpus tpu-vm ssh "
            f"{self.name} "
            f"--zone {self.zone} "
            f"--command "
        )
        command = command.split()
        for env_stmt in env_stmts:
            cmd = env_stmt + cmd
        command.append(cmd)
        stdout, stderr, retcode = execute(command, run_async=run_async, capture_output=capture_output)
        return stdout

    @property
    def ip_address(self):
        command = f"gcloud compute tpus describe {self.name} --zone {self.zone} --format=value(networkEndpoints[0].ipAddress)"
        stdout, stderr, retcode = execute(command.split(), capture_output=True)
        return stdout.strip()

    def clean_up(self):
        self.ssh("pkill -9 python3")

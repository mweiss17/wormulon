import torch
import time
from wormulon.core import Node
from wormulon.fncall import FunctionCall
from wormulon.utils import execute
from wormulon.tpu_handler import TPUHandler


class TPU(Node):
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
        self.bucket = bucket
        self._job_handlers = []

    def submit(self, experiment_directory: str, fn: FunctionCall, *args, **kwargs):
        handler = TPUHandler.instantiate(
            bucket=self.bucket,
            experiment_directory=experiment_directory,
            function_call=FunctionCall(fn, args, kwargs),
        )
        self._job_handlers.append(handler)
        return handler.launch(self)

    def delete(self):
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
              --version tpu-vm-pt-1.10"

            if self.preemptible:
                command += " --preemptible"

            output = execute(command.split())
            breakpoint()
            if output.returncode == 0:
                return output
            else:
                if retry:
                    time.sleep(10)
                else:
                    return output
        return output

    def ssh(self, cmd, env_stmts=[], synchronous=True, timeout=30):
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
        print(f"running: {command} on {self.name}")

        output = execute(command, timeout=timeout)
        return output

    @property
    def internal_ip(self):
        command = f"gcloud compute tpus describe {self.name} --zone {self.zone} --format=value(networkInterfaces[0].networkIP)"
        output = execute(command.split())
        return output.stdout.decode("utf-8").strip()

    def clean_up(self):
        self.ssh("pkill -9 python3")

    @property
    def last_heartbeat(self):
        data = torch.load(self.bucket.download(self.prefix + "heartbeat.pt"))
        return data["last_heartbeat"]

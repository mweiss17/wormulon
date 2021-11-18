import torch
import time
from wormulon.core import Node
from wormulon.fncall import FunctionCall
from wormulon.utils import execute
from wormulon.tpu_handler import TPUHandler


class TPU(Node):
    def __init__(
        self, zone, network, subnet, netrange, acc_type, preemptible, bucket, name=None
    ):
        self.zone = zone
        self.network = network
        self.subnet = subnet
        self.netrange = netrange
        self.acc_type = acc_type
        self.preemptible = preemptible
        self.bucket = bucket
        self._wandb_api_key = None

        if name is not None:
            self.name = name
        else:
            node_ids, error = self.get_tpu_ids()
            self.name = f"node-{max(node_ids) + 1}"
            self.create()

    def submit(self, fn: FunctionCall, *args, **kwargs):
        handler = JobHandler.instantiate(
            executor_directory=self.executor_directory,
            function_call=FunctionCall(fn, args, kwargs),
        )
        self._job_handlers.append(handler)
        return handler.launch(rc=self.rc, verbose=self.verbose,)

    def get_tpu_ids(self):
        command = f"gcloud alpha compute tpus list --zone={self.zone} --format=value[seperator=','](name)"
        output, error = execute(command.split())
        ids = output.decode("utf-8").split("\n")
        ids.remove("")
        int_ids = [-1]
        int_ids.extend([int(i.split("-")[-1]) for i in ids])
        return int_ids, error

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

            output, error = execute(command.split())
            print(f"error was: {error}")
            if error is not None:
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

    def ssh(self, cmd, use_env=True, synchronous=True, timeout=30):
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

        output, error = execute(command, synchronous=synchronous, timeout=timeout)
        return output, error

    @property
    def internal_ip(self):
        command = f"gcloud compute tpus describe {self.name} --zone {self.zone} --format=value(networkInterfaces[0].networkIP)"
        output, error = execute(command.split())
        return output.decode("utf-8").strip()

    def clean_up(self):
        self.ssh("pkill -9 python3")

    @property
    def last_heartbeat(self):
        data = torch.load(self.bucket.download(self.prefix + "heartbeat.pt"))
        return data["last_heartbeat"]

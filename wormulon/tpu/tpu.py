import torch
import time
import asyncio
from wormulon.tpu.fncall import FunctionCall
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

    @property
    def is_ready(self):
        command = f"gcloud compute tpus list --format=value(NAME,STATUS) --zone {self.zone}"
        stdout, stderr, retcode = execute(command.split(), capture_output=True)
        rows = stdout.split("\n")
        rows.remove("")
        names = {r.split("\t")[0] for r in rows if r.split("\t")[1] == "READY"}
        return self.name in names

    def delete(self):
        print(f"deleting tpu {self.name}")
        return execute(
            f"gcloud alpha compute tpus tpu-vm delete {self.name} --zone {self.zone} --async --quiet".split()
        )

    def create(self):
        while True:
            command = f"gcloud alpha compute tpus tpu-vm create {self.name} \
              --zone {self.zone} \
              --network {self.network} \
              --subnetwork {self.subnet} \
              --range {self.netrange} \
              --accelerator-type {self.acc_type} \
              --version tpu-vm-pt-1.10 \
               {'--preemptible' if self.preemptible else ''}"
            command = command.split()
            command.append("--metadata")
            command.append("shutdown-script=\'#! /bin/bash;"
                           ""
                           " pgrep -f python3 | xargs kill -SIGTERM;"
                           " while pgrep -f python3 > /dev/null;"
                           " do sleep 1; "
                           "done; \'")
            print(command)
            stdout, stderr, retcode = execute(command, capture_output=True)
            if retcode == 0:
                return stdout, retcode
            else:
                return stderr, retcode

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
        if run_async:
            return stdout
        return stdout, stderr, retcode

    @property
    def ip_address(self):
        command = f"gcloud compute tpus describe {self.name} --zone {self.zone} --format=value(networkEndpoints[0].ipAddress)"
        stdout, stderr, retcode = execute(command.split(), capture_output=True)
        return stdout.strip()

    def clean_up(self):
        self.ssh("pkill -9 python3")

    def __eq__(self, other):
        return self is not None and other is not None and self.name == other.name
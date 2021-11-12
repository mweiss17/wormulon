#!/usr/bin/env python
import time
import subprocess
from typing import List
from wormulon.utils import _upload_data_to_gcs


def execute(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def get_node_ids(zone="us-central1-f") -> List[int]:
    command = f"gcloud alpha compute tpus list --zone={zone} --format=value[seperator=','](name)"
    output, error = execute(command.split())
    ids = output.decode("utf-8").split("\n")
    ids.remove("")
    int_ids = [int(i.split("-")[-1]) for i in ids]
    return int_ids, error


def create_tpu_node(
    node_id,
    network,
    subnetwork,
    network_range,
    tpu_type,
    preemptible=False,
    asynch=False,
):

    node_name = f"node-{node_id}"
    print(f"creating: {node_name}")

    command = f"gcloud alpha compute tpus tpu-vm create {node_name} \
      --zone us-central1-f \
      --network {network} \
      --subnetwork {subnetwork} \
      --range {network_range} \
      --accelerator-type {tpu_type} \
      --version v2-alpha"

    if preemptible:
        command += " --preemptible"
    if asynch:
        command += " --async"

    return execute(command.split())


def run_ssh_cmd_on_tpu_node(node_id, cmd):

    node_name = f"node-{node_id}"
    command = (
        f"gcloud alpha compute tpus tpu-vm ssh "
        f"{node_name} "
        f"--zone us-central1-f "
        f"--command "
    )

    command = command.split()
    command.append(cmd)

    print(f"running: {command} on {node_name}")

    output, error = execute(command)
    return output, error


def get_internal_ip(node_id, zone="us-central1-f"):
    command = f"gcloud compute tpus describe node-{node_id} --zone {zone} --format=value(networkInterfaces[0].networkIP)"
    output, error = execute(command.split())
    return output.decode("utf-8").strip()


def delete_tpu_node(node_id, zone="us-central1-f"):
    command = f"gcloud compute tpus tpu-vm delete node-{node_id} --zone {zone} --async"
    output, error = execute(command.split())
    return output, error


def check_preempted(node_id):
    command = f"gcloud compute operations list --filter=operationType=compute.instances.preempted"
    # command = f"gcloud compute tpus describe node-{node_id} --format=value(status)"
    output, error = execute(command.split())
    # Poll and check for pre-empted. If pre-empted, then
    # curl -sfH 'Metadata-Flavor: Google' 'http://169.254.169.254/computeMetadata/v1/instance/preempted'
    return output


def tpu_submit(
    bucket,
    experiment_directory,
    trainer,
    training_state,
    network,
    subnetwork,
    network_range,
    tpu_type,
    preemptible,
    **kwargs,
):
    node_ids, error = get_node_ids()
    node_id = max(node_ids) + 1
    prefix = f"/{experiment_directory}/{str(time.time())}"
    trainer_path = f"{prefix}/trainer.pt"
    training_state_path = f"{prefix}/training_state.pt"
    print(f"uploading training state and trainer to {prefix}.")
    _upload_data_to_gcs(bucket, trainer_path, trainer.serialize())
    _upload_data_to_gcs(bucket, training_state_path, training_state.serialize())

    print(f"Currently, we have tpu nodes: {node_ids}, adding {node_id} now...")

    create_tpu_node(node_id, network, subnetwork, network_range, tpu_type, preemptible)

    cmd = (
        f"cd ~/; git clone https://github.com/mweiss17/polytax.git; "
        f"pip install -e polytax[xla]; "
    )
    output, error = run_ssh_cmd_on_tpu_node(node_id, cmd)

    cmd = f"cd ~/polytax/src/polytax; python3 train.py {bucket} {trainer_path} {training_state_path} "
    output, error = run_ssh_cmd_on_tpu_node(node_id, cmd)


#
# @click.command(
#     context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
# )
# @click.argument("root_id", default=1, type=int)
# @click.argument("num_nodes", default=1, type=int)
# @click.argument("exp_name", default="experiments/t5-xs-shakespeare", type=str)
# @click.argument("template_name", default="experiments/t5-xs-shakespeare", type=str)
# @click.argument("network", default="tpu-network", type=str)
# @click.argument("subnetwork", default="swarm-2", type=str)
# @click.argument("network_range", default="192.169.0.0/29", type=str)
# @click.argument("tpu_type", default="v2-8", type=str)
# @click.option("-p", "--preemptible", is_flag=True, default=False)
# def tpu_submit(
#     root_id,
#     num_nodes,
#     exp_name,
#     template_name,
#     network,
#     subnetwork,
#     network_range,
#     tpu_type,
#     preemptible,
#     **kwargs,
# ):
#     command = f"gcloud alpha compute tpus tpu-vm create node-{root_id} \
#       --zone us-central1-f \
#       --network {network} \
#       --subnetwork {subnetwork} \
#       --range {network_range} \
#       --accelerator-type {tpu_type} \
#       --version v2-alpha"
#     if preemptible:
#         command += f" --preemptible"
#
#     execute(command.split())
#
#     # Get internal-ip for the control plane node
#     command = (
#         f"gcloud alpha compute tpus describe node-{root_id} --format=get(ipAddress)"
#     )
#     output, error = execute(command.split())
#
#     # Boot all the other nodes (if booting more than one
#     for i in range(1, num_nodes):
#         node_name = f"node-{i+root_id}"
#         print(f"creating: {node_name}")
#         # Spool up a TPU node
#         command = f"gcloud alpha compute tpus tpu-vm create {node_name} \
#           --zone us-central1-f \
#           --network {network} \
#           --subnetwork {subnetwork} \
#           --range {range} \
#           --accelerator-type {tpu_type} \
#           --version v2-alpha \
#           --async"
#         if preemptible:
#             command += " --preemptible"
#
#         execute(command.split())
#
#     # ssh into each node and setup polytax
#     for i in range(0, num_nodes):
#         node_name = f"node-{i+root_id}"
#         print(f"running start script on: {node_name}")
#         command = (
#             f"gcloud alpha compute tpus tpu-vm ssh "
#             f"{node_name} "
#             f"--zone us-central1-f "
#             f"--command "
#         )
#         command = command.split()
#         command.append(
#             f"cd ~/; git clone https://github.com/mweiss17/polytax.git; chmod 755 polytax/src/polytax/scripts/*.sh; ./polytax/src/polytax/scripts/launch_tpu.sh {i} localhost {exp_name} {template_name}"
#         )
#         output, error = execute(command)

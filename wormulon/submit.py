#!/usr/bin/env python
import os
import sys
import click
import submitit
from submitit.core.utils import CommandFunction

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument('train_script')
@click.argument('experiment_dir')
@click.option('--mem_gb')
@click.option('--cpus_per_task')
@click.option('--slurm_gres')
def main(train_script, experiment_dir, **kwargs):

    # create the submitit executor for creating and managing jobs
    ex = submitit.AutoExecutor(folder=os.path.join(experiment_dir, "Logs"))

    # setup the executor parameters based on the cluster location
    if ex.cluster == "slurm":
        ex.update_parameters(
            mem_gb=kwargs.get("mem_gb") or 8,
            cpus_per_task=kwargs.get("cpus_per_task") or 12,
            timeout_min=kwargs.get("timeout_min") or 1000,
            tasks_per_node=1,
            nodes=1,
            slurm_partition=kwargs.get("slurm_partition") or "long",
            slurm_gres=kwargs.get("slurm_gres"),
        )
    command = ["python3"] + sys.argv[1:]
    print(f"running: {command}")

    ex.submit(CommandFunction(command))

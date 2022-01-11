import os
import sys
import click
import wandb
import pickle
from pathlib import Path
from wormulon.tpu.tpu_manager import TPUManager
from wormulon.tpu.tpu_job import TPUJob
from wormulon.train_state import TrainState
from wormulon.tpu.nanny import Nanny



def submit_job(train_script, train_cls, *args):
    exec(f"from {train_script} import {train_cls}")
    sys.argv = [train_script] + list(args)
    trainer = eval(train_cls)()
    jobs = []
    for i in range(trainer.get("distributed/kwargs/world_size")):

        print(f"creating job-{i}")
        job = TPUJob(trainer)
        job.write_to_disk()
        jobs.append(job)

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def main(kwargs):
    submit_job(kwargs.args[0], kwargs.args[1], *kwargs.args[2:])

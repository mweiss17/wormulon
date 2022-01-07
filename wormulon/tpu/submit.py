import sys
import click
import asyncio
import pickle
from wormulon.tpu.nanny import Nanny
from pathlib import Path
import os
import sys
import wandb
import signal
import click
import asyncio
from wormulon.tpu.tpu_manager import TPUManager
from wormulon.tpu.tpu_job import TPUJob
from wormulon.train_state import TrainState

def submit_job(train_script, train_cls, *args):
    exec(f"from {train_script} import {train_cls}")
    sys.argv = [train_script] + list(args)
    trainer = eval(train_cls)()
    jobs = []
    for i in range(trainer.get("distributed/kwargs/world_size")):
        print(f"creating job-{i}")
        job = TPUJob(trainer)
        jobs.append(job)
        pickle.dump(job,
                    open(f"{trainer.experiment_directory}/Logs/job-{job.trainer.get('distributed/kwargs/rank')}.pkl",
                         "wb"))

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def main(kwargs):
    submit_job(kwargs.args[0], kwargs.args[1], *kwargs.args[2:])

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


class TPUSubmitter:

    def __init__(self):
        super(TPUSubmitter, self).__init__()

    def create_jobs(self, trainer):
        manager = TPUManager(**trainer.get("tpu/kwargs"))
        tpus = manager.get_tpus(trainer.get("distributed/kwargs/world_size"))
        jobs = []
        for i, tpu in enumerate(tpus):
            print(f"Dibs on TPU: {tpu}, creating job-{i}")
            env_stmts = trainer.get("job/kwargs/env_stmts")
            env_stmts.append(f"export WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')};")
            trainer.set('job/kwargs/env_stmts', env_stmts)
            if len(tpus) > 1:
                trainer.set("distributed/kwargs/init_method", f"tcp://{tpus[0].ip_address}:2345")
            trainer.set('distributed/kwargs/rank', i)
            job = TPUJob(trainer, tpu)
            pickle.dump(job, open(f"{trainer.experiment_directory}/Logs/job-{job.trainer.get('distributed/kwargs/rank')}.pkl", "wb"))


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.argument("train_script")
@click.argument("train_cls")
@click.option("--mem_gb")
@click.option("--cpus_per_task")
@click.option("--slurm_gres")
def main(train_script, train_cls, **kwargs):
    exec(f"from {train_script} import {train_cls}")
    sys.argv = [sys.argv[0]] + sys.argv[3:]
    trainer = eval(train_cls)()
    TPUSubmitter().create_jobs(trainer)

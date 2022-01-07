import os
import sys
import wandb
import pickle
import signal
import click
import asyncio
from collections import defaultdict
from wormulon.tpu.tpu_manager import TPUManager
from wormulon.tpu.tpu_job import TPUJob
from wormulon.train_state import TrainState
from wormulon.utils import JobState

class Nanny:

    def __init__(self, experiment_directory):
        super(Nanny, self).__init__()
        self.jobs = defaultdict(set)
        self.experiment_directory = experiment_directory
        self.managers = {}
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.exit_gracefully)

    def setup_wandb(self, trainer):
        wandb_run = wandb.init(name=trainer.wandb_run_name, job_type=trainer.WANDB_JOB_TYPE, dir=trainer.wandb_directory,
                               resume=False, project=trainer.WANDB_PROJECT, config=trainer.wandb_config, entity=trainer.WANDB_ENTITY)
        wandb.finish()
        return wandb_run.id

    def find_jobs(self):
        for job_dir in os.listdir(self.experiment_directory):
            for log_file in os.listdir(f"{self.experiment_directory}/{job_dir}/Logs"):
                if log_file.startswith("job-"):
                    job_file = f"{self.experiment_directory}/{job_dir}/Logs/{log_file}"
                    job = pickle.load(open(job_file, "rb"))
                    self.jobs[job_dir].add(job)

    async def launch_jobs(self):
        for job_dir, jobs in self.jobs.items():
            for job_ix, job in enumerate(jobs):
                if job.future is not None and job.status is not JobState.PREEMPTED:
                    continue
                zone = job.trainer.get("tpu/kwargs/bucket")
                if self.managers.get(zone) is None:
                    self.managers[zone] = TPUManager(**job.trainer.get("tpu/kwargs"))
                tpus = self.managers[zone].get_tpus(job.trainer.get("distributed/kwargs/world_size"))
                print(f"Launching job-{job_ix} {job} on TPU: {tpus[job_ix]}")

                env_stmts = job.trainer.get("job/kwargs/env_stmts")
                env_stmts.append(f"export WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')};")
                job.trainer.set('job/kwargs/env_stmts', env_stmts)
                if len(tpus) > 1:
                    trainer.set("distributed/kwargs/init_method", f"tcp://{tpus[0].ip_address}:2345")
                job.trainer.set('distributed/kwargs/rank', job_ix)
                train_state = TrainState.initial_state(step=0, epoch=0, misc_attributes={"wandb_run_id": self.setup_wandb(job.trainer)})

                success = job.arm(train_state, tpus[job_ix], resume=True)
                if not success:
                    print(f"Failed to launch job-{job_ix} {job} on TPU: {tpus[job_ix]}")
                    continue
                future = job.submit()
                job.future = future

    async def monitor_jobs(self, verbose=True):
        for job_dir, jobs in self.jobs.items():
            for job in jobs:
                with open(f"{job.trainer.experiment_directory}/Logs/submission.txt", "w") as f:
                    print(f"monitoring job-{job.trainer.get('distributed/kwargs/rank')}: {job_dir}, status: {job.status}")

                    if job.status is not JobState.RUNNING:
                        print(f"{job} is not in RUNNING state")
                        continue

                    out = job.outbuffer.getvalue()
                    err = job.errbuffer.getvalue()
                    if out:
                        out = f"job-{job.trainer.get('distributed/kwargs/rank')}: {out}"
                        if verbose:
                            print(out)
                        f.write(out)
                        job.outbuffer.seek(0)
                        job.outbuffer.truncate()
                    if err:
                        err = f"job-{job.trainer.get('distributed/kwargs/rank')}: {err}"
                        if verbose:
                            print(err)
                        f.write(err)
                        job.errbuffer.seek(0)
                        job.errbuffer.truncate()

        await asyncio.sleep(1)

    async def run(self):
        while True:
            # Finds jobs on disk and adds them to the job set.
            self.find_jobs()
            # Launches jobs that are not already running.
            await self.launch_jobs()
            # Waits for jobs to finish.
            await self.monitor_jobs()


    def exit_gracefully(self, signum, frame):
        print("Exiting gracefully")
        for job_dir, jobs in self.jobs.items():
            for job in jobs:
                job.clean_up()
                print(f"{job.tpu} is now available")
                print(f"exited {job.train_state.misc_attributes.get('wandb_run_url')}")
        sys.exit(0)


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.argument("experiment_directory")
def main(experiment_directory, **kwargs):
    nanny = Nanny(experiment_directory)
    asyncio.run(nanny.run())

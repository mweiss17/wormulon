import os
import sys
import wandb
import time
import pickle
import signal
import glob
import sys
import wandb
import time
import pickle
import signal
import click
from multiprocessing import Process
from collections import defaultdict
from wormulon.tpu.tpu_manager import TPUManager
from wormulon.tpu.tpu_job import TPUJob
from wormulon.utils import JobState

class Nanny:

    def __init__(self, experiment_directory):
        super(Nanny, self).__init__()
        self.jobs = dict()
        self.managers = dict()
        self.job_procs = dict()
        self.experiment_directory = experiment_directory
        signal.signal(signal.SIGINT, self.exit_gracefully)

    def write_to_logfile(self, message, verbose=False):
        if verbose:
            print(message, flush=True)
        with open("nanny.txt", "a") as fp:
            fp.write(f"{message}\n")

    def find_jobs(self):
        """ Looks through the experiments directory and finds jobs. Adds them to dictionary if they aren't already there."""
        for job_path in glob.glob(f"{self.experiment_directory}/*/Logs/*.pkl"):
            job = pickle.load(open(job_path, "rb"))
            if job.job_id not in self.jobs.keys():
                self.jobs[job.job_id] = job

    def get_tpu(self, job):
        """ Helper function that builds a TPU manager if there isn't one, then finds (or creates) a TPU for the job"""
        zone = job.trainer.get("tpu/kwargs/bucket")
        if self.managers.get(zone) is None:
            self.managers[zone] = TPUManager(**job.trainer.get("tpu/kwargs"))
        tpu = self.managers[zone].get_tpus(job.trainer.get("distributed/kwargs/world_size"))[0]
        return tpu

    def add_wandb_api_key(self, job):
        env_stmts = job.trainer.get("job/kwargs/env_stmts")
        env_stmts.add(f"export WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')};")
        job.trainer.set('job/kwargs/env_stmts', env_stmts)
        os.environ["WANDB_SILENT"] = "True"

    def setup_job(self, job):
        """ Builds the job, adding a TPU, wandb key """
        tpu = self.get_tpu(job)
        job.set_tpu(tpu)
        self.add_wandb_api_key(job)

    def launch_jobs(self):
        for job_id, job in self.jobs.items():
            if job.status in {JobState.ARMED, JobState.RUNNING, JobState.SUCCESS}:
                continue
            self.write_to_logfile(f"Launching job {job}")
            self.setup_job(job)
            job_proc = Process(target=job.launch, daemon=False)
            job_proc.start()
            self.job_procs[job_id] = job_proc

    def cleanup(self):
        # iterates over job_procs that have died
        for job_id, job in self.jobs.items():
            job_proc = self.job_procs.get(job_id)
            if job_proc is not None and not job_proc.is_alive():
                self.write_to_logfile(f"Nanny is cleaning up the dead proc: {job}.")
                job.clean_up()
                del self.job_procs[job_id]
            breakpoint()

            # Checks heartbeats on storage
            # TODO: Implement this


    def run(self):
        while True:
            self.find_jobs()
            self.launch_jobs()
            self.cleanup()
            self.write_to_logfile([job for job in self.jobs.values()])
            time.sleep(5)

    def exit_gracefully(self, signum, frame):
        self.write_to_logfile("Exiting gracefully")
        for job_id, job in self.jobs.items():
            job_proc = self.job_procs.get(job_id)
            if job_proc is not None:
                job_proc.terminate()
            job.clean_up()
        sys.exit(0)


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.argument("experiment_directory")
def main(experiment_directory, **kwargs):
    nanny = Nanny(experiment_directory)
    nanny.run()

import os
import sys
import wandb
import time
import pickle
import signal
import os
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


    def find_jobs(self):
        for job_path in glob.glob(f"{self.experiment_directory}/*/Logs/*.pkl"):
            job = pickle.load(open(job_path, "rb"))
            self.jobs[job.job_id] = job

    def setup_job(self, job, job_ix=0):
        zone = job.trainer.get("tpu/kwargs/bucket")
        if self.managers.get(zone) is None:
            self.managers[zone] = TPUManager(**job.trainer.get("tpu/kwargs"))
        tpu = self.managers[zone].get_tpus(job.trainer.get("distributed/kwargs/world_size"))[job_ix]
        job.set_tpu(tpu)
        env_stmts = job.trainer.get("job/kwargs/env_stmts")
        env_stmts.append(f"export WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')};")
        job.trainer.set('job/kwargs/env_stmts', env_stmts)
        job.trainer.set('distributed/kwargs/rank', job_ix)
        # if len(tpus) > 1:
        #     trainer.set("distributed/kwargs/init_method", f"tcp://{tpus[0].ip_address}:2345")


    def launch_jobs(self):
        for job_ix, job in enumerate(self.jobs.values()):
            if job.status in {JobState.ARMED, JobState.RUNNING, JobState.SUCCESS}:
                continue
            self.setup_job(job)
            if self.job_procs.get(job.job_id) is not None:
                self.job_procs[job.job_id].terminate()
            job_proc = Process(target=job.launch, daemon=False)
            job_proc.start()
            self.job_procs[job.job_id] = job_proc

    def run(self):
        while True:
            self.find_jobs()
            self.launch_jobs()
            time.sleep(5)

    def exit_gracefully(self, signum, frame):
        print("Exiting gracefully")
        for job_id, job in self.jobs.items():
            self.job_procs[job_id].terminate()
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
    nanny.run()

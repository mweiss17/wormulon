import os
import asyncio
import datetime
import pickle
import time
import wandb
from wormulon.core import Job
from wormulon.utils import dump_yaml, load_yaml
from wormulon.utils import JobState
from wormulon.train_state import TrainState
from wormulon.tpu.bucket import Bucket
from wormulon.tpu.fncall import FunctionCall

class TPUJob(Job):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.bucket = Bucket(trainer.get("tpu/kwargs/bucket"))
        self.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.STARTING.value}), overwrite=True)
        self.created_at = datetime.datetime.now()
        self.tpu = None
        self.train_state = None

    @property
    def logfile_path(self):
        return os.path.join(self.trainer.experiment_directory, "Logs", "job-log.txt")

    @property
    def errfile_path(self):
        return os.path.join(self.trainer.experiment_directory, "Logs", "job-err.txt")

    def write_to_logfile(self, message, verbose=False):
        if verbose:
            print(message, flush=True)
        with open(self.logfile_path, "a") as fp:
            fp.write(f"{message}\n")

    def write_to_errfile(self, message):
        with open(self.errfile_path, "a") as fp:
            fp.write(f"{message}\n")

    @property
    def name(self):
        return self.trainer.experiment_directory.split("/")[-1]

    @property
    def remote_working_directory(self):
        path = os.path.join(self.trainer.experiment_directory, self.job_id)
        return path

    @property
    def function_call_serialization_path(self):
        return os.path.join(self.remote_working_directory, "function_call.pkl")

    @property
    def env(self):
        return self.trainer.get("job/kwargs/env_stmts")

    @property
    def install(self):
        return self.trainer.get("job/kwargs/install_cmd")

    @property
    def train(self):
        return self.trainer.get("job/kwargs/train_cmd")

    @property
    def setup(self):
        return self.trainer.get("job/kwargs/setup_cmds")

    @property
    def cleanup(self):
        return self.trainer.get("job/kwargs/cleanup_cmd")

    @property
    def job_state_path(self):
        return os.path.join(self.remote_working_directory, "jobstate.yml")

    @property
    def failed(self):
        return ExceptionInJob.is_instance(self.function_call.outputs) or JobFailure.is_instance(
            self.function_call.outputs
        )

    def nonblocking_ssh(self, cmd, env):
        proc = self.tpu.ssh(cmd, env, run_async=True)

        while True:
            curtime = time.strftime('%X')
            out = proc.stdout.read()
            err = proc.stderr.read()
            out = out.decode("utf-8") if out else ""
            err = err.decode("utf-8") if err else ""
            if out != "":
                out = f"{self.name}, {self.tpu}, {curtime}, job-{self.trainer.get('distributed/kwargs/rank')}: {out}"
                self.write_to_logfile(out)
            if err != "":
                err = f"{self.name}, {self.tpu}, {curtime}, job-{self.trainer.get('distributed/kwargs/rank')}: {err}"
                self.write_to_logfile(err)
            poll = proc.poll()
            if poll is None:
                time.sleep(1)
            elif "Finished worker" in out:
                return True
            else:
                return poll

    @property
    def local_pickle_path(self):
        return f"{self.trainer.experiment_directory}/Logs/job-{self.trainer.get('distributed/kwargs/rank')}.pkl"

    def write_to_disk(self):
        with open(self.local_pickle_path, "wb") as fp:
            pickle.dump(self, fp)

    def update_train_state(self):
        self.train_state = TrainState.initial_state(step=0, epoch=0,
                                               misc_attributes={"wandb_run_id": self.setup_wandb()})
        self.write_to_logfile("New train state initialized to step 0.")

    def setup_wandb(self):
        wandb_run = wandb.init(name=self.trainer.wandb_run_name, job_type=self.trainer.WANDB_JOB_TYPE,
                               dir=self.trainer.wandb_directory,
                               resume=False, project=self.trainer.WANDB_PROJECT, config=self.trainer.wandb_config,
                               entity=self.trainer.WANDB_ENTITY)
        wandb.finish()
        return wandb_run.id

    @property
    def is_alive(self):
        data = self.bucket.get_blob(self.trainer.experiment_directory + "/heartbeat")

        # No heartbeat file means the job hasn't started yet
        if data is None:
            return True

        if data.updated > self.last_heartbeat + datetime.timedelta(seconds=30):
            return False

        self.last_heartbeat = data.updated
        return True

    def clean_up(self):
        name = ""
        if self.tpu is not None:
            name = self.tpu.name
        self.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.FAILURE.value, "tpu_name": name}), overwrite=True)
        if self.tpu:
            print(f"{self.tpu} is now available")
        if self.train_state is not None:
            print(f"exited {self.train_state.misc_attributes.get('wandb_run_url')}")

    @property
    def status(self):
        try:
            state = load_yaml(self.bucket.download(self.job_state_path).getvalue())['state']
        except Exception as e:
            self.write_to_logfile(e)
            state = JobState.UNKNOWN.value
        return JobState(state)

    def set_tpu(self, tpu):
        self.tpu = tpu
        self.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.ARMED.value, "tpu_name": self.tpu.name}), overwrite=True)

    def arm(self):
        self.write_to_logfile("Arming job")

        if not self.tpu.is_ready:
            stderr, retcode = self.tpu.create()
            if retcode != 0:
                return False

        return True

    def launch(self):
        self.update_train_state()
        success = self.arm()
        if not success:
            self.write_to_logfile(f"Failed to launch {self} on TPU: {self.tpu}")
            return None
        self.write_to_logfile("cleanup")
        self.tpu.ssh(self.cleanup, self.env, check=False)
        self.function_call = FunctionCall(self.trainer, self.train_state, self.trainer.get("job/kwargs"), self.tpu.name)
        self.bucket.upload(self.function_call_serialization_path, self.function_call.serialize(), overwrite=True)

        # setup the TPU
        for cmd in self.setup:
            self.write_to_logfile(f"Running setup command: {cmd}")
            _, _, retcode = self.tpu.ssh(cmd, self.env, capture_output=True)
            # If we need to install everything we get an error and then do it here
            if retcode == 1:
                self.write_to_logfile(f"Installing {self.install}")
                _, _, retcode = self.tpu.ssh(self.install, self.env, run_async=False, capture_output=True)
                if retcode != 0:
                    self.clean_up()
                    raise Exception(f"Failed to install {self.install}")
                break
        self.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.RUNNING.value, "tpu_name": self.tpu.name}), overwrite=True)
        train_cmd = f"{self.train} {self.bucket.name} {self.remote_working_directory}"
        self.write_to_logfile(f"Running train command: {train_cmd}")
        self.nonblocking_ssh(train_cmd, self.env)
        return self.function_call.outputs

    def __eq__(self, other):
        return self.job_id == other.job_id and self.created_at == other.created_at

    def __hash__(self):
        return hash(self.job_id)

    def __repr__(self):
        return f"{self.name}, {self.tpu}, {self.created_at}"
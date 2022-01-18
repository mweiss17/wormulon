import os
import sys
import signal
import click
from wormulon.tpu.bucket import Bucket
from wormulon.tpu.fncall import FunctionCall
from wormulon.train_state import TrainState
from wormulon.utils import dump_yaml, JobState
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index, fn_call_buffer, bucket_name, job_state_path):
    print(f"Starting worker {index}", flush=True)
    fn_call = FunctionCall.deserialize(fn_call_buffer)
    bucket = Bucket(bucket_name)
    try:
        train_state = bucket.get_latest_trainstate(fn_call.trainer.experiment_directory)
    except IndexError as e:
        print(f"{e}. Failed to get_latest_trainstate, getting one on the functioncall", flush=True)
        trainstate_buf = bucket.download(fn_call.trainstate)
        train_state = TrainState.deserialize(trainstate_buf)
    fn_call.trainstate = train_state
    fn_call.call()

    if fn_call.trainstate.step >= fn_call.trainer.get("num_train_steps") and index == 0:
        bucket.upload(job_state_path, dump_yaml({"state": JobState.SUCCESS.value, "tpu_name": fn_call.tpu_name}), overwrite=True)
    print(f"Finished worker {index} with output: {fn_call.outputs}", flush=True)
    sys.exit(0)


class JobRunner(object):
    def __init__(self, bucket_name, directory):
        self.bucket = Bucket(bucket_name)
        self.directory = directory
        original_sigint = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    @property
    def fn_call_path(self):
        path = os.path.join(self.directory, "function_call.pkl")
        return path

    @property
    def job_state_path(self):
        return os.path.join(self.directory, "jobstate.yml")

    @property
    def trainstate_path(self):
        return os.path.join(self.directory, "trainstate.pkl")

    def run(self):
        fn_call_buffer = self.bucket.download(self.fn_call_path)
        xmp.spawn(_mp_fn, args=(fn_call_buffer.getvalue(), self.bucket.name, self.job_state_path), nprocs=8, daemon=False, start_method="fork")

    def exit_gracefully(self, signum, frame):
        print("Job is exiting gracefully")
        self.bucket.upload(self.job_state_path, dump_yaml({"state": JobState.PREEMPTED.value}), overwrite=True)
        sys.exit(0)

@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_trainstate=True)
)
@click.argument("bucket_name")
@click.argument("directory")
def main(bucket_name, directory):
    JobRunner(bucket_name, directory).run()

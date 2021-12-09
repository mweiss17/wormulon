import os
import argparse
import torch
from wormulon.tpu.bucket import Bucket
from wormulon.utils import ExceptionInJob, JobFailure, deserialize
from wormulon.tpu.fncall import FunctionCall
from wormulon.train_state import TrainState
import torch_xla.distributed.xla_multiprocessing as xmp
import click

def _mp_fn(index, fn_call_buffer, bucket_name):
    print(f"Starting worker {index}")
    fn_call = FunctionCall.deserialize(fn_call_buffer)
    if type(fn_call.trainstate) == str:
        trainstate_buf = Bucket(bucket_name).download(fn_call.trainstate)
        fn_call.trainstate = TrainState.deserialize(trainstate_buf)
    fn_call.call()
    print(f"Finished worker {index}")


class JobRunner(object):
    def __init__(self, bucket_name, directory):
        self.bucket = Bucket(bucket_name)
        self.directory = directory

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
        xmp.spawn(_mp_fn, args=(fn_call_buffer.getvalue(), self.bucket.name), nprocs=8, start_method="fork")


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_trainstate=True)
)
@click.argument("bucket_name")
@click.argument("directory")
def main(bucket_name, directory):
    JobRunner(bucket_name, directory).run()

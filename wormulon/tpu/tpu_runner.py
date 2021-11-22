import os
import argparse
import torch
from wormulon.tpu.bucket import Bucket
from wormulon.utils import ExceptionInJob, JobFailure
from wormulon.tpu.fncall import FunctionCall
import torch_xla.distributed.xla_multiprocessing as xmp
import click


class JobRunner(object):
    def __init__(self, bucket_name, directory):
        self.bucket = Bucket(bucket_name)
        self.directory = directory

    @property
    def fn_call_path(self):
        path = os.path.join(self.directory, "function_call.pt")
        return path

    @property
    def job_state_path(self):
        return os.path.join(self.directory, "job_state.yml")

    @property
    def trainstate_path(self):
        return os.path.join(self.directory, "trainstate.pt")

    @property
    def function_output_serialization_path(self):
        return os.path.join(self.working_directory, "function_output.pt")

    def print_pre_exit_info(self, function_call: FunctionCall):
        if isinstance(function_call.outputs, ExceptionInJob):
            print(
                f"FunctionCall failed due to the following exception:"
                f"\n{function_call.outputs.exception}"
            )
        elif isinstance(function_call.outputs, JobFailure):
            print(
                "FunctionCall failed due to not being able to assign "
                "output from the function. "
            )
        else:
            print(
                f"FunctionCall returned an output of type "
                f"{type(function_call.outputs)} at "
                f"{self.function_output_serialization_path}."
            )
        return self

    def run(self):
        def _mp_fn(index, bucket, fn_call_buffer, path):
            fn_call = torch.load(fn_call_buffer)

            fn_call.call()
            fn_call.serialize_outputs(bucket, path)

        fn_call_buffer = self.bucket.download(self.fn_call_path)

        xmp.spawn(
            _mp_fn, args=(self.bucket, fn_call_buffer, self.function_output_serialization_path), nprocs=8, start_method="fork",
        )


    def heartbeat(self):
        heartbeat_path = os.path.join(self.directory, "heartbeat")
        self.bucket.touch(heartbeat_path)


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.argument("bucket_name")
@click.argument("directory")
def main(bucket_name, directory):
    JobRunner(bucket_name, directory).run()

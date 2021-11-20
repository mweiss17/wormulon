import os
import argparse
import torch
from wormulon.bucket import Bucket
from wormulon.utils import ExceptionInJob, JobFailure
from wormulon.fncall import FunctionCall
import torch_xla.distributed.xla_multiprocessing as xmp


class JobRunner(object):
    def __init__(self, bucket_name, directory):
        self.bucket = Bucket(bucket_name)
        self.directory = directory
        self.run()

    @property
    def fn_call_path(self):
        path = os.path.join(self.directory, "function_call.pt")
        return path

    @property
    def job_status_path(self):
        return os.path.join(self.directory, "JobState.yml")

    @property
    def trainstate_path(self):
        return os.path.join(self.directory, "trainstate.pkl")

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
        def _mp_fn(index, bucket, trainer_buffer, trainstate_buffer):
            trainer = torch.load(trainer_buffer)
            trainstate = torch.load(trainstate_buffer)
            while trainstate.total_steps < trainstate.steps:
                trainstate = trainer(trainstate)
                if index == 0:
                    bucket.upload(trainstate)
            trainer.finish()

        trainer_buffer = self.bucket.download(self.fn_call_path)
        trainstate_buffer = self.bucket.download(self.trainstate_path)

        xmp.spawn(
            _mp_fn,
            args=(self.bucket, trainer_buffer, trainstate_buffer),
            nprocs=8,
            start_method="fork",
        )

        # function_call = FunctionCall.deserialize(
        #     path=self.function_call_serialization_path,
        # )
        # Call the function
        # function_call.call()
        # Accquire the file lock and serialize
        # function_call.serialize_outputs(path=self.function_output_serialization_path)
        # Print and exit
        # self.print_pre_exit_info(trainer)

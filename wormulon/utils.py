import io
import os
import pathlib
import yaml
import subprocess
from dataclasses import dataclass
from addict import Dict
from enum import Enum

class JobState(Enum):
    RUNNING = 0
    SUCCESS = 1
    FAILURE = 2
    ABORTED = 3
    TIMEOUT = 4
    STARTING = 5
    PREEMPTED = 6


def execute(command, capture_output=False, run_async=False):
    output = ("", "", 0)
    try:
        if run_async:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.set_blocking(proc.stdout.fileno(), False)
            os.set_blocking(proc.stderr.fileno(), False)
            return proc, None, None
        elif capture_output:
            output = subprocess.run(
                command, capture_output=capture_output, timeout=300, check=True
            )
            output = output.stdout.decode("utf-8"), output.stderr.decode("utf-8"), output.returncode
        else:
            subprocess.run(command, check=True)
    except subprocess.TimeoutExpired as e:
        print(
            f"command failed with {e}, taking longer than 300 seconds to finish."
        )
    except subprocess.CalledProcessError as e:
        print(f"command failed with exit code {e}")
    return output


class NotAvailable(object):
    pass


def is_instance(obj, cls):
    # This is required because python's native isinstance might fail on
    # deserialized objects.
    return obj.__class__.__name__ == cls.__name__


@dataclass
class ExceptionInJob(object):
    exception: str

    @classmethod
    def is_instance(cls, obj):
        return is_instance(obj, cls)


class JobFailure(object):
    @classmethod
    def is_instance(cls, obj):
        return is_instance(obj, cls)


class JobTimeout(object):
    @classmethod
    def is_instance(cls, obj):
        return is_instance(obj, cls)


def serialize(object_to_serialize):
    try:
        import torch_xla.core.xla_model as xm
    except Exception:
        import torch

        xm = None

    buffer = io.BytesIO()
    if xm:
        xm.save(object_to_serialize, buffer)
    else:
        torch.save(object_to_serialize, buffer)
    return buffer.getvalue()


def deserialize(buffer):
    try:
        import torch_xla
    except Exception:
        import torch

        torch_xla = None

    if torch_xla:
        ob = torch_xla.utils.serialization.load(buffer)
    else:
        ob = torch.load(buffer)
    return ob


def dump_yaml(d: dict):
    return yaml.dump(d)

def load_yaml(buffer: bytes):
    return Dict(yaml.load(buffer, Loader=yaml.FullLoader))


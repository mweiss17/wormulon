import io
import yaml
import subprocess
from dataclasses import dataclass
from addict import Dict


class JobState:
    RUNNING = 0
    SUCCESS = 1
    FAILURE = 2
    ABORTED = 3
    TIMEOUT = 4
    STARTING = 5
    PREEMPTED = 6

    @staticmethod
    def to_string(status):
        if status == JobState.RUNNING:
            return "RUNNING"
        elif status == JobState.SUCCESS:
            return "SUCCESS"
        elif status == JobState.FAILURE:
            return "FAILURE"
        elif status == JobState.ABORTED:
            return "ABORTED"
        else:
            return "UNKNOWN"


def execute(command, timeout=300, capture_output=True):
    try:
        output = subprocess.run(
            command, capture_output=capture_output, timeout=timeout, check=True
        )
        return output.stdout.decode("utf-8"), output.stderr.decode("utf-8"), 0

    except subprocess.TimeoutExpired as e:
        print(
            f"command failed with {e}, taking longer than {timeout} seconds to finish."
        )
    except subprocess.CalledProcessError as e:
        print(f"command failed with exit code {e.returncode}, {e.stderr}")
    return ("", "", 1)


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
        import torch_xla.core.xla_model as xm
    except Exception:
        import torch

        xm = None

    if xm:
        ob = xm.load(buffer)
    else:
        ob = torch.load(buffer)
    return ob


def dump_yaml(d: dict, path: str):
    with open(path, "w") as f:
        yaml.dump(d, f)


def load_yaml(path: str):
    with open(path, "r") as f:
        return Dict(yaml.load(f, Loader=yaml.FullLoader))

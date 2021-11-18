import io
import yaml
import subprocess
from dataclasses import dataclass
from addict import Dict


class JobStatus:
    RUNNING = 0
    SUCCESS = 1
    FAILURE = 2
    ABORTED = 3
    TIMEOUT = 4
    STARTING = 5
    PREEMPTED = 6

    @staticmethod
    def to_string(status):
        if status == JobStatus.RUNNING:
            return "RUNNING"
        elif status == JobStatus.SUCCESS:
            return "SUCCESS"
        elif status == JobStatus.FAILURE:
            return "FAILURE"
        elif status == JobStatus.ABORTED:
            return "ABORTED"
        else:
            return "UNKNOWN"


def execute(command, synchronous=True, timeout=30):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = None, None

    if synchronous:
        output, error = process.communicate()
    else:
        try:
            output, error = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"command took longer than {timeout} seconds to finish. Continuing.")
    return output, error


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

import io
import traceback
import pickle
import stopit
from typing import Callable, Union, Any
from dataclasses import dataclass, field
from contextlib import suppress
from wormulon.utils import (
    NotAvailable,
    ExceptionInJob,
    JobFailure,
    JobTimeout,
    serialize,
)


@dataclass
class FunctionCall(object):
    trainer: Callable
    trainstate: Any
    kwargs: dict
    outputs: Union[Any, NotAvailable] = NotAvailable()
    timeout: int = 2628000

    def call(self):
        if not isinstance(self.outputs, NotAvailable):
            # This is to ensure that the function is called only once.
            return self.outputs

        try:
            self.outputs = self.trainer(self.trainstate)
        except Exception as e:
            traceback.print_exc()
            print(e)
            self.outputs = ExceptionInJob(traceback.format_exc())
        finally:
            if isinstance(self.outputs, NotAvailable):
                self.outputs = JobFailure()
        return self

    def serialize(self):
        buffer = pickle.dumps((self.trainer, self.trainstate, self.kwargs))
        return buffer

    @classmethod
    def deserialize(cls, buffer):
        return cls(*pickle.loads(buffer))

    def serialize_outputs(self):
        return serialize(self.outputs)
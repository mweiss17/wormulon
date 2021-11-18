import traceback
import stopit
from typing import Callable, Union, Any, field,
from dataclasses import dataclass
from contextlib import suppress
from wormulon.utils import (
    NotAvailable,
    ExceptionInJob,
    JobFailure,
    JobTimeout,
)


@dataclass
class FunctionCall(object):
    fn: Callable
    args: tuple
    kwargs: dict
    outputs: Union[Any, NotAvailable] = NotAvailable()

    def call(self):
        if not isinstance(self.outputs, NotAvailable):
            # This is to ensure that the function is called only once.
            return self.outputs
        if self.configuration.get("timeout") is not None:
            timer = stopit.ThreadingTimeout(
                self.configuration.get("timeout"), swallow_exc=False
            )
        else:
            timer = suppress()
        try:
            with timer:
                self.outputs = self.fn(*self.args, **self.kwargs)
        except stopit.TimeoutException:
            self.outputs = JobTimeout()
        except Exception:
            self.outputs = ExceptionInJob(traceback.format_exc())
        finally:
            if isinstance(self.outputs, NotAvailable):
                self.outputs = JobFailure()
        return self

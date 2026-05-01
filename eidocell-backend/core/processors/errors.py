"""Typed errors raised by processors.

Services should distinguish:
- UnknownProcessorError: bad method name → HTTP 400 at the router boundary.
- ProcessorInputError: bad input (shape/dtype/empty mask) → per-sample skip with logged reason,
  or HTTP 400 if it applies to the whole input.
- ProcessorRuntimeError: model/library failure on otherwise-valid input → per-sample skip, logged.

Anything else (genuine bugs) bubbles up untouched.
"""


class ProcessorError(Exception):
    """Base for processor-raised errors."""


class UnknownProcessorError(ProcessorError):
    """Requested method is not registered."""


class ProcessorInputError(ProcessorError):
    """Input fails the processor's preconditions (shape, dtype, missing data)."""


class ProcessorRuntimeError(ProcessorError):
    """Processor failed at runtime on otherwise-valid-looking input."""

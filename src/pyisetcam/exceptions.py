"""Package exceptions."""

from __future__ import annotations


class PyISETCamError(Exception):
    """Base package error."""


class MissingAssetError(PyISETCamError):
    """Raised when a required upstream asset is missing."""


class OctaveExecutionError(PyISETCamError):
    """Raised when an Octave parity command cannot complete."""

    def __init__(
        self,
        message: str,
        *,
        command: list[str] | None = None,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        crash_log: str | None = None,
    ) -> None:
        super().__init__(message)
        self.command = command or []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.crash_log = crash_log


class ExplicitObjectRequiredError(PyISETCamError):
    """Raised when a MATLAB-style implicit object lookup would be required."""


class UnsupportedOptionError(NotImplementedError):
    """Raised for milestone-one unsupported options."""

    def __init__(self, matlab_function: str, option: str) -> None:
        super().__init__(f"{matlab_function} does not support option '{option}' in milestone one.")
        self.matlab_function = matlab_function
        self.option = option

"""The exceptions of tdmspy. Everything the package raises on purpose is a `TdmsError`."""


class TdmsError(Exception):
    """Base class of every error raised for a file's content."""


class TdmsFormatError(TdmsError, ValueError):
    """The file is not valid TDMS (wrong signature, corrupt or inconsistent segments, ...)."""


class UnsupportedFeatureError(TdmsError, NotImplementedError):
    """The file is valid TDMS but uses something this package does not read (DAQmx raw data, ...)."""


class UnsupportedScalingError(UnsupportedFeatureError):
    """A channel's NI scaling uses a scale type that is not implemented (`read(scaled=False)` still works)."""

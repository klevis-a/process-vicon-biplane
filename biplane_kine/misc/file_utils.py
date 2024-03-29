"""This module contains various utilities for dealing with files."""

from typing import Tuple, Any

# bytes pretty-printing
UNITS_MAPPING = (
    (1 << 50, ' PB'),
    (1 << 40, ' TB'),
    (1 << 30, ' GB'),
    (1 << 20, ' MB'),
    (1 << 10, ' KB'),
    (1, (' byte', ' bytes')),
)


# retrieved from https://stackoverflow.com/a/12912296/2577053
def pretty_size(num_bytes: float, units: Tuple[Any] = UNITS_MAPPING) -> str:
    """Get human-readable file sizes.
    simplified version of https://pypi.python.org/pypi/hurry.filesize/
    """
    factor = None
    suffix = None
    for factor, suffix in units:
        if num_bytes >= factor:
            break
    amount = int(num_bytes / factor)

    if isinstance(suffix, tuple):
        singular, multiple = suffix
        if amount == 1:
            suffix = singular
        else:
            suffix = multiple
    return str(amount) + suffix

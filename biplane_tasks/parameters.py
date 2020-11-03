from pathlib import Path
from typing import Union, Dict, Any
import json


def read_smoothing_exceptions(file_name: Union[str, Path]) -> Dict[str, Any]:
    """Read the smoothing exceptions file and create a nested dictionary from it."""
    all_exceptions = {}
    if file_name:
        with open(file_name) as f:
            all_exceptions = json.load(f)
    return all_exceptions


def marker_smoothing_exceptions(all_exceptions: Dict[str, Any], trial_name: str, marker_name: str) -> Dict[str, Any]:
    """Given all exceptions (all_exceptions) return just the ones for the specified marker (marker_name) and trial
    (trial_name)."""
    trial_exceptions = all_exceptions.get(trial_name, {})
    return trial_exceptions.get(marker_name, {})


def smoothing_exceptions_for_marker(file_name: Union[str, Path], trial_name: str, marker_name: str):
    """Read the smoothing exceptions file and return the exception for the specified marker (marker_name) and trial
    (trial_name)."""
    all_exceptions = read_smoothing_exceptions(file_name)
    return marker_smoothing_exceptions(all_exceptions, trial_name, marker_name)


def read_filling_directives(file_name: Union[str, Path]) -> Dict[str, Any]:
    """Read the filling directives file and create a nested dictionary from it."""
    return read_smoothing_exceptions(file_name)


def trial_filling_directives(all_directives: Dict[str, any], trial_name: str) -> Dict[str, any]:
    """Give all directives (all_directives) return just the ones for the specified trial (trial_name)."""
    return all_directives.get(trial_name, {})


def filling_directives_for_trial(file_name: Union[str, Path], trial_name: str) -> Dict[str, any]:
    """Read the filling directives file and return just the ones for the specified trial (trial_name)."""
    return trial_filling_directives(read_filling_directives(file_name), trial_name)

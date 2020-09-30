import json


def read_smoothing_exceptions(file_name):
    all_exceptions = {}
    if file_name:
        with open(file_name) as f:
            all_exceptions = json.load(f)
    return all_exceptions


def marker_smoothing_exceptions(all_exceptions, trial_name, marker_name):
    trial_exceptions = all_exceptions.get(trial_name, {})
    return trial_exceptions.get(marker_name, {})


def smoothing_exceptions_for_marker(file_name, trial_name, marker_name):
    all_exceptions = read_smoothing_exceptions(file_name)
    return marker_smoothing_exceptions(all_exceptions, trial_name, marker_name)

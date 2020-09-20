from misc.json_utils import json2obj
import json

class Params:
    params = None

    # note that this only works for one file as it stands, but it's easy enough to make params a dictionary to make it
    # work for multiple files
    @classmethod
    def get_params(cls, params_file):
        if Params.params is None:
            Params.params = json2obj(params_file)
        return Params.params


def read_smoothing_exceptions(file_name):
    all_exceptions = {}
    if file_name:
        with open(file_name) as f:
            all_exceptions = json.load(f)
    return all_exceptions


def marker_smoothing_exceptions(all_exceptions, trial_name, marker_name):
    trial_exceptions = all_exceptions.get(trial_name, {})
    return trial_exceptions.get(marker_name, {})

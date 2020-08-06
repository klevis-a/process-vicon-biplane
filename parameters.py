from misc.json_utils import json2obj


class Params:
    params = None

    # note that this only works for one file as it stands, but it's easy enough to make params a dictionary to make it
    # work for multiple files
    @classmethod
    def get_params(cls, params_file):
        if Params.params is None:
            Params.params = json2obj(params_file)
        return Params.params

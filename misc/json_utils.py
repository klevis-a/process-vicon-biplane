import json
from collections import namedtuple


def _json_object_hook(d):
    return namedtuple('params', d.keys())(*d.values())


def json2obj(file):
    with open(file) as f:
        return json.load(f, object_hook=_json_object_hook)

"""This module contains various useful generic Python utilities."""
import functools
from typing import Callable, Any


class NestedDescriptor:
    """A class that is a descriptor, which allows access to items in the nested container via the supplied
    get_item_method."""
    def __init__(self, nested_container: Any, get_item_method: Callable):
        self._nested_container = nested_container
        self._get_item_method = get_item_method

    def __getitem__(self, item: Any) -> Any:
        return self._get_item_method(self._nested_container, item)


def rgetattr(obj, attr, *args):
    """Return the nested attribute specified in attr."""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

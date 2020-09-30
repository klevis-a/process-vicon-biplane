import functools


def partialclass(cls, *args, **kwds):
    class NewClass(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)
    return NewClass


class NestedContainer:
    def __init__(self, nested_container, get_item_method):
        self._nested_container = nested_container
        self._get_item_method = get_item_method

    def __getitem__(self, item):
        return self._get_item_method(self._nested_container, item)

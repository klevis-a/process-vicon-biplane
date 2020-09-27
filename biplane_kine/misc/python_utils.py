import functools


def partialclass(cls, *args, **kwds):
    class NewClass(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)
    return NewClass

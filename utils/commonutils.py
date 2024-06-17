from dill.source import getsource
from dill import detect


def function_to_string(fn):
    return getsource(detect.code(fn))

from typing import Callable
from os.path import exists, join, dirname
from os import makedirs


def created_parent_dir(pathfun: Callable) -> str:
    def make_parent_and_return(*args, **kwargs):
        path = pathfun(*args, **kwargs)
        if not exists(dirname(path)):
            makedirs(dirname(path))
        return path

    return make_parent_and_return


def created_dir(pathfun: Callable) -> str:
    def make_parent_and_return(*args, **kwargs):
        path = pathfun(*args, **kwargs)
        if not exists(path):
            makedirs(path)
        return path

    return make_parent_and_return


@created_parent_dir
def fjoin(*args) -> str:
    return join(*args)


@created_dir
def djoin(*args) -> str:
    return join(*args)

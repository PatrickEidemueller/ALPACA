from typing import Callable, Generator
from copy import copy
import random
import concurrent.futures

from alpaca.system.interfaces.dataset import Dataset, DataLoaderInterface

_THREAD_POOL_EXECUTOR = concurrent.futures.ThreadPoolExecutor()


def load_batches(
    dataset: "Dataset",
    batchsize: int,
    shuffle: bool = True,
    transform: Callable = None,
    preload: int = 32,
) -> Generator:
    """
    Splits the datset into batches and handles asynchronous pre-loading.

    @param batchsize : Size of each batch
    @param shuffle : Whether to load the data randomly shuffled
    """

    assert preload > 1
    lengths = [batchsize] * int(len(dataset) / batchsize)
    remaining = len(dataset) - len(lengths) * batchsize
    if remaining > 0:
        lengths.append(remaining)

    ids = dataset.ids
    if shuffle:
        ids = copy(ids)
        random.shuffle(ids)

    partial_sum = 0
    batches = []
    for l in lengths:
        batches.append(ids[partial_sum : partial_sum + l])
        partial_sum += l

    load_fun = (
        dataset.data_loader.load_batch
        if transform is None
        else lambda b: transform(dataset.data_loader.load_batch(b))
    )

    futures = []
    futures = [_THREAD_POOL_EXECUTOR.submit(load_fun, b) for b in batches[:preload]]

    n_yielded = 0
    while n_yielded < len(batches):
        yield futures[n_yielded].result()
        if len(futures) < len(batches):
            futures.append(
                _THREAD_POOL_EXECUTOR.submit(load_fun, batches[n_yielded + preload])
            )
        n_yielded += 1
    assert all(f.done() for f in futures)

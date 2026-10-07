from __future__ import annotations

import os
import resource
from contextlib import contextmanager
from typing import Iterator


MAX_THREADS = 12
MAX_MEMORY_GB = 12
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def configure_process_limits(*, threads: int, memory_gb: int) -> None:
    """Apply hard ceilings before analysis work starts."""
    max_threads = int(os.environ.get("TABR1_MAX_THREADS", MAX_THREADS))
    max_memory_gb = int(os.environ.get("TABR1_MAX_MEMORY_GB", MAX_MEMORY_GB))
    if not 1 <= threads <= max_threads:
        raise ValueError(f"threads must be between 1 and {max_threads}, got {threads}")
    if not 1 <= memory_gb <= max_memory_gb:
        raise ValueError(f"memory_gb must be between 1 and {max_memory_gb}, got {memory_gb}")

    for name in THREAD_ENV_VARS:
        os.environ[name] = str(threads)

    memory_bytes = memory_gb * 1024**3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = memory_bytes if soft == resource.RLIM_INFINITY else min(soft, memory_bytes)
    new_hard = memory_bytes if hard == resource.RLIM_INFINITY else min(hard, memory_bytes)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))


@contextmanager
def thread_limit(threads: int) -> Iterator[None]:
    """Limit native thread pools already loaded by NumPy/scikit-learn."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=threads):
        yield

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Coroutine, TypeVar

T = TypeVar("T")


def run_coroutine(coro: Coroutine[None, None, T]) -> T:
    """Run async code from sync contexts, including nested Jupyter event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()

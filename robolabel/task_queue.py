import asyncio
import asyncio
from typing import Callable, Any
import logging


def as_async_task(coro: Callable[..., Any]) -> Callable[..., asyncio.Task[Any] | None]:
    def async_wrapper(*args: Any, **kwargs: Any) -> asyncio.Task[Any] | None:
        if asyncio.get_event_loop().is_running():
            task = asyncio.create_task(coro(*args, **kwargs), name="exclusive_robolabel_task")
        else:
            task = asyncio.get_event_loop().run_until_complete(coro(*args, **kwargs))
        return task

    return async_wrapper

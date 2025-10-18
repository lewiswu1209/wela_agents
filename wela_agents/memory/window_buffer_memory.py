
from typing import TypeVar

from wela_agents.memory.buffer_memory import BufferMemory

T = TypeVar("T")

class WindowBufferMemory(BufferMemory[T]):

    def __init__(self, memory_key: str, window_size: int) -> None:
        super().__init__(memory_key)
        self._window_size: int = window_size

    def save_content(self, content: T) -> None:
        buffer = self.get_contents(None)
        buffer.append(content)
        self._buffer = buffer[-self._window_size:]

__all__ = [
    "WindowBufferMemory"
]


from typing import List
from typing import TypeVar

from wela_agents.memory.memory import Memory

T = TypeVar("T")

class BufferMemory(Memory[T]):

    def __init__(self, memory_key: str) -> None:
        super().__init__(memory_key)
        self._buffer: List[T] = []

    def save_content(self, content: T) -> None:
        self._buffer.append(content)

    def get_contents(self, _: List[T]) -> List[T]:
        return self._buffer

    def reset_memory(self) -> None:
        self._buffer.clear()

__all__ = [
    "BufferMemory"
]

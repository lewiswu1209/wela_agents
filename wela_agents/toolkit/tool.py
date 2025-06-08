

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List

class Tool(ABC):
    def __init__(self, name: str, description: str, required: List[str], **properties: Any) -> None:
        self.name: str = name
        self.description: str = description
        self.param_description: Dict = properties
        self.required: List[str] = required

    @abstractmethod
    def _invoke(self, **kwargs: Any) -> str:
        pass

    def run(self, **kwargs: Any) -> str:
        result = self._invoke(**kwargs)
        return result if result else ""

    def to_tool_param(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.param_description,
                    "required": self.required,
                },
            }
        }

__all__ = [
    "Tool"
]

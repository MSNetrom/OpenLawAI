"""Base class for chat modes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, Dict, Union

from agents.models import (
    ChatHistory,
    ModeName,
    ModeResult,
    QualityModeName,
    StreamEvent,
)

if TYPE_CHECKING:
    from chat_manager import ChatManager


class Mode(ABC):
    """Base class for chat modes. Yields StreamEvents and returns ModeResult."""
    name: ModeName
    models: Dict[QualityModeName, str] = {}  # Override in subclasses

    def get_model(self, chat_history: ChatHistory) -> str:
        """Get the model to use based on quality mode setting."""
        quality_mode: QualityModeName = chat_history.metadata.quality_mode
        return self.models[quality_mode]

    @abstractmethod
    async def run(self, manager: "ChatManager", chat_history: ChatHistory) -> AsyncIterator[Union[StreamEvent, ModeResult]]:
        """Run the mode, yielding stream events. Final yield must be ModeResult."""
        raise NotImplementedError
        yield  # Make this a generator



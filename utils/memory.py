"""
Multi-Turn Summary Buffer Memory Utility.
Manages multi-turn conversation history by maintaining a sliding window of recent turns
alongside an aggregated summary of earlier conversation turns to prevent prompt context token overflow.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ConversationSummaryBuffer:
    """Sliding Summary Buffer Memory for session conversation history."""
    
    def __init__(self, max_recent_turns: int = 4):
        self.max_recent_turns = max_recent_turns
        self.history: List[Dict[str, str]] = []
        self.summary: str = ""
        
    def add_turn(self, role: str, content: str):
        """Adds a new message turn to history."""
        self.history.append({"role": role, "content": content})
        self._update_summary_if_needed()
        
    def _update_summary_if_needed(self):
        """Compresses older turns into a rolling summary when history exceeds window limit."""
        if len(self.history) > self.max_recent_turns:
            turns_to_summarize = self.history[:-self.max_recent_turns]
            summary_snippets = []
            for item in turns_to_summarize:
                role = item["role"].capitalize()
                summary_snippets.append(f"{role}: {item['content'][:100]}...")
                
            self.summary = f"Summary of earlier turns: {' | '.join(summary_snippets)}"
            # Retain only recent turns
            self.history = self.history[-self.max_recent_turns:]
            logger.info(f"Compressed older chat turns into summary buffer.")

    def get_formatted_context(self) -> str:
        """Returns formatted string combining summary and recent history."""
        parts = []
        if self.summary:
            parts.append(f"[{self.summary}]")
            
        for item in self.history:
            role = item["role"].capitalize()
            parts.append(f"{role}: {item['content']}")
            
        return "\n".join(parts)

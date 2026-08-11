"""
Guardrails & Prompt Injection Classifier Utility.
Scans incoming user prompts for adversarial jailbreaks, system prompt leakage attempts,
and malicious injection attacks before passing them to the LLM agent.
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Heuristic adversarial pattern regex rules
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt\s+leak",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"you\s+are\s+now\s+in\s+dan\s+mode",
    r"jailbreak",
    r"override\s+security\s+rules",
    r"exec\s*\(",
    r"eval\s*\(",
    r"import\s+os",
    r"import\s+sys",
    r"subprocess",
    r"rm\s+-rf"
]

def validate_query_safety(prompt: str) -> Tuple[bool, str]:
    """
    Evaluates natural language query for security compliance and safety.
    
    Args:
        prompt (str): Natural language user prompt.
        
    Returns:
        Tuple[bool, str]: (is_safe, violation_reason)
    """
    if not prompt or not prompt.strip():
        return False, "Query cannot be empty."
        
    prompt_lower = prompt.lower()
    
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, prompt_lower):
            logger.warning(f"Guardrails flagged security violation matching pattern: '{pattern}'")
            return False, "Security Violation: Potential prompt injection or system override attempt detected."
            
    return True, "Query passed security guardrails check."

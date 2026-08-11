"""
Multi-Agent Delegation Pipeline Utility.
Orchestrates specialized sub-agent roles (Data Analyst, Visualization Specialist, Executive Summarizer)
to route user queries to the optimal specialized prompt strategy.
"""
import logging
from typing import Dict, Any, Tuple


logger = logging.getLogger(__name__)

class AgentTeamOrchestrator:
    """Orchestrates query classification and specialization routing for sub-agents."""
    
    @staticmethod
    def classify_intent(prompt: str) -> str:
        """
        Classifies user prompt intent into agent specialization roles:
        - 'VISUALIZATION'
        - 'STATISTICAL_ANALYSIS'
        - 'GENERAL_EXECUTIVE_SUMMARY'
        """
        prompt_lower = prompt.lower()
        
        chart_keywords = ["plot", "chart", "graph", "histogram", "scatter", "bar chart", "line chart", "visualize", "draw"]
        if any(kw in prompt_lower for kw in chart_keywords):
            return "VISUALIZATION"
            
        stats_keywords = ["summary", "mean", "median", "std", "correlation", "t-test", "anova", "outliers", "p-value", "highest", "lowest", "sum"]
        if any(kw in prompt_lower for kw in stats_keywords):
            return "STATISTICAL_ANALYSIS"
            
        return "GENERAL_EXECUTIVE_SUMMARY"

    @classmethod
    def enhance_prompt_for_role(cls, prompt: str) -> Tuple[str, str]:
        """
        Routes user query and returns (role_name, enhanced_prompt).
        """
        role = cls.classify_intent(prompt)
        
        if role == "VISUALIZATION":
            enhanced = f"[Role: Visualization Specialist Agent] Focus on generating concise Plotly code for: {prompt}"
        elif role == "STATISTICAL_ANALYSIS":
            enhanced = f"[Role: Data Analyst Agent] Focus on exact statistical metrics and dataframe aggregation for: {prompt}"
        else:
            enhanced = f"[Role: Executive Summarizer Agent] Focus on high-level business takeaways for: {prompt}"
            
        logger.info(f"Agent Team Orchestrator routed query to role '{role}'")
        return role, enhanced

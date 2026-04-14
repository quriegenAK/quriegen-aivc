"""agents/ — AIVC agent layer (v1).

Wraps AgentDispatcher with deterministic + Claude-backed agents.
See docs/aivc_spec_agents_v1.md.
"""
from agents.base_agent import AgentResult, AgentTask, BaseAgent, MCPTool
from agents.data_agent import DataAgent, DataReport
from agents.eval_agent import EvalAgent
from agents.research_agent import ResearchAgent
from agents.training_agent import TrainingAgent

__all__ = [
    "AgentResult",
    "AgentTask",
    "BaseAgent",
    "MCPTool",
    "DataAgent",
    "DataReport",
    "EvalAgent",
    "ResearchAgent",
    "TrainingAgent",
]

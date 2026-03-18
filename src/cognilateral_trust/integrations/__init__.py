"""Framework integrations for the Cognilateral Trust Engine.

Optional integrations for popular agent frameworks. Each integration
is a thin wrapper around evaluate_trust() — zero additional dependencies
beyond the framework itself.

Available integrations:
    - langgraph: TrustNode for LangGraph StateGraph workflows
    - crewai: TrustTool for CrewAI agent tool chains
"""

from __future__ import annotations

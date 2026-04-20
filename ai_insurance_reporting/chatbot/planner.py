"""Rule-based planner for selecting chatbot tools."""

from __future__ import annotations


class AgentPlanner:
    """Rule-based tool planner designed for later replacement."""

    EXECUTION_VERBS = {"rerun", "reruns", "run", "refresh", "rebuild", "regenerate", "execute"}
    EXECUTION_TARGETS = {
        "workflow",
        "pipeline",
        "validation",
        "forecast",
        "forecasting",
        "explainability",
        "narrative",
        "visualization",
        "visualizations",
        "chart",
        "charts",
        "figure",
        "figures",
        "chatbot",
        "index",
        "retrieval",
        "data",
        "etl",
    }

    KEYWORD_MAP = {
        "ScenarioRunTool": {"scenario", "stress", "stressed", "shock", "sensitivity"},
        "ScenarioSummaryTool": {"scenario", "stressed", "adverse", "sensitive", "sensitivity", "impact"},
        "ValidationSummaryTool": {"validation", "issue", "issues", "anomaly", "anomalies", "quality", "reconciliation", "material"},
        "ForecastComparisonTool": {"forecast", "forecasting", "model", "outlook", "performed", "actual", "comparison", "capital", "reserve", "claims", "premium", "largest", "changed", "segment", "q3", "q4"},
        "MovementAnalysisTool": {"movement", "bridge", "waterfall", "opening", "closing", "beginning", "ending", "step", "steps"},
        "ExplainabilityTool": {"driver", "drivers", "explain", "why", "shap", "lime", "pdp", "ice", "factor", "factors"},
        "NarrativeLookupTool": {"summary", "outlook", "commentary", "narrative", "management", "draft"},
        "FigureLookupTool": {"figure", "figures", "chart", "charts", "plot", "visual", "visualization"},
        "RunSummaryTool": {"run", "artifact", "artifacts", "stage", "stages", "governance", "completed"},
        "AnalystReviewTool": {"review", "triage", "priority", "pending", "queue", "recommended", "action", "actions"},
    }

    def plan(self, question: str) -> list[str]:
        question_lower = question.lower()
        tokens = self._tokenize(question)
        selected: list[str] = []
        if "what if" in question_lower:
            selected.append("ScenarioRunTool")
        if (tokens & self.EXECUTION_VERBS) and (
            tokens & self.EXECUTION_TARGETS or any(phrase in question_lower for phrase in {"full workflow", "entire workflow", "whole workflow"})
        ):
            selected.append("WorkflowExecutionTool")
        for tool_name, keywords in self.KEYWORD_MAP.items():
            if tokens & keywords:
                selected.append(tool_name)

        if not selected:
            return []

        ordered_tools = [
            "WorkflowExecutionTool",
            "ScenarioRunTool",
            "ScenarioSummaryTool",
            "ValidationSummaryTool",
            "ForecastComparisonTool",
            "MovementAnalysisTool",
            "ExplainabilityTool",
            "NarrativeLookupTool",
            "FigureLookupTool",
            "RunSummaryTool",
            "AnalystReviewTool",
        ]
        return [tool for tool in ordered_tools if tool in selected]

    def _tokenize(self, text: str) -> set[str]:
        cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
        return {token for token in cleaned.split() if token}


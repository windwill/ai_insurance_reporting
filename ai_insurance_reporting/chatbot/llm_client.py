"""Pluggable LLM clients for the chatbot."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
import json
from typing import Callable, Protocol
from urllib import error, request


class LLMClient(Protocol):
    """Protocol for pluggable LLM generation."""

    def generate(self, prompt: str) -> str:
        """Generate an answer from a prompt."""


@dataclass(slots=True)
class MockLLMClient:
    """Deterministic local fallback for grounded answer generation."""

    def generate(self, prompt: str) -> str:
        tool_output_text = self._extract_block(prompt, "Tool outputs:", "Retrieved context:")
        if tool_output_text and tool_output_text.strip() not in {"{}", ""}:
            tool_summary = self._summarize_tool_outputs(tool_output_text)
            if tool_summary:
                return tool_summary

        context_items = self._parse_context(prompt)
        if not context_items:
            return "The reporting artifacts do not contain enough information to answer this question."

        management_mode = "assisting senior management reviewing an insurance reporting package" in prompt.lower()
        primary = context_items[0]
        primary_content = primary["content"].rstrip(".")
        answer_parts: list[str] = []

        if management_mode:
            answer_parts.append(f"Management view: {primary_content}.")
        else:
            answer_parts.append(primary_content + ".")

        if len(context_items) > 1:
            secondary = context_items[1]["content"].rstrip(".")
            connector = self._choose_connector(primary_content, secondary, management_mode=management_mode)
            answer_parts.append(f"{connector} {secondary}.")

        if len(context_items) > 2:
            tertiary = context_items[2]["content"].rstrip(".")
            if management_mode:
                answer_parts.append(
                    f"From a management perspective, supporting evidence indicates that {self._lowercase_first(tertiary)}."
                )
            else:
                answer_parts.append(f"Supporting evidence also indicates that {self._lowercase_first(tertiary)}.")

        return " ".join(answer_parts)

    def _parse_context(self, prompt: str) -> list[dict[str, str]]:
        context_lines = [line.strip() for line in prompt.splitlines() if line.strip().startswith("[")]
        items: list[dict[str, str]] = []
        pattern = re.compile(
            r"^\[(?P<document_id>[^\]]+)\]\s+"
            r"(?P<title>.*?)\s+\|\s+dataset=(?P<dataset>.*?)\s+\|\s+"
            r"filters=(?P<filters>.*?)\s+\|\s+content=(?P<content>.*)$"
        )
        for line in context_lines:
            match = pattern.match(line)
            if not match:
                continue
            items.append(match.groupdict())
        return items

    def _extract_block(self, prompt: str, start_marker: str, end_marker: str) -> str:
        if start_marker not in prompt:
            return ""
        start = prompt.index(start_marker) + len(start_marker)
        end = prompt.index(end_marker) if end_marker in prompt[start:] else len(prompt)
        return prompt[start:end].strip()

    def _summarize_tool_outputs(self, tool_output_text: str) -> str:
        try:
            import json

            payload = json.loads(tool_output_text)
        except Exception:
            return ""

        if not payload:
            return ""

        parts: list[str] = []
        if "WorkflowExecutionTool" in payload:
            workflow = payload["WorkflowExecutionTool"]
            mode = workflow.get("mode", "workflow")
            requested = workflow.get("requested_stage_labels") or workflow.get("requested_agents") or []
            executed = workflow.get("executed_stage_labels") or workflow.get("executed_agents") or []
            if requested:
                if mode == "full":
                    parts.append("Management view: The full reporting workflow was rerun successfully.")
                else:
                    parts.append(
                        f"Management view: The requested workflow stages were rerun successfully for {', '.join(requested)}."
                    )
            if executed:
                parts.append(f"Executed stages included {', '.join(executed)}.")
            governance_log_path = workflow.get("governance_log_path", "")
            if governance_log_path:
                parts.append(f"Updated governance output was written to {governance_log_path}.")

        if "ScenarioRunTool" in payload:
            scenario = payload["ScenarioRunTool"]
            summary = scenario.get("summary") or {}
            comparison = scenario.get("comparison") or {}
            deltas = comparison.get("comparison_metrics") or []
            biggest_delta = max(deltas, key=lambda item: abs(item.get("change_pct", 0.0))) if deltas else {}
            if summary:
                parts.append(
                    f"Management view: Scenario {scenario.get('scenario_name', 'scenario')} reran successfully through "
                    f"{summary.get('latest_quarter', 'the latest quarter')} with a synthetic capital-to-liability proxy ratio of "
                    f"{summary.get('capital_ratio', 'n/a')}."
                )
            if biggest_delta:
                parts.append(
                    f"The largest balance sheet movement versus baseline is in {biggest_delta.get('metric_name', 'the selected metric')}, "
                    f"changing by {biggest_delta.get('change', 'n/a')} or {biggest_delta.get('change_pct', 'n/a')}."
                )
            forecast_deltas = comparison.get("forecast_comparison") or []
            if forecast_deltas:
                capital_outlook = next(
                    (item for item in forecast_deltas if item.get("metric_name") == "capital_ratio"),
                    forecast_deltas[0],
                )
                parts.append(
                    f"Scenario forecast comparisons indicate {self._metric_label(capital_outlook.get('metric_name', 'the outlook metric'))} "
                    f"moves from {capital_outlook.get('baseline_value', 'n/a')} to {capital_outlook.get('scenario_value', 'n/a')}."
                )

        if "ForecastComparisonTool" in payload:
            forecast = payload["ForecastComparisonTool"]
            best_model = forecast.get("best_model") or {}
            selected_target = forecast.get("selected_target") or "the selected metric"
            if best_model:
                parts.append(
                    f"Management view: For {self._metric_label(selected_target)}, the best-performing model is "
                    f"{best_model.get('model_family', 'the available model')} with MAE {best_model.get('mae', 'n/a')} "
                    f"and RMSE {best_model.get('rmse', 'n/a')}."
                )
            forecast_rows = forecast.get("forecast_rows") or []
            if forecast_rows:
                first = forecast_rows[0]
                parts.append(
                    f"The forward outlook indicates {self._metric_label(selected_target)} of {first.get('forecast_value', 'n/a')} "
                    f"for {first.get('forecast_quarter', 'the next quarter')}."
                )

        if "ValidationSummaryTool" in payload:
            validation = payload["ValidationSummaryTool"]
            latest = (validation.get("validation_summary") or [{}])[0]
            if latest:
                parts.append(
                    f"Control results for {latest.get('quarter', validation.get('latest_quarter', 'the latest quarter'))} "
                    f"show a validation pass rate of {latest.get('validation_pass_rate', 'n/a')} with "
                    f"{latest.get('records_with_issues', 'n/a')} records with issues and "
                    f"{latest.get('anomaly_count', validation.get('anomaly_count_total', 'n/a'))} anomalies."
                )

        if "ExplainabilityTool" in payload:
            explain = payload["ExplainabilityTool"]
            drivers = explain.get("shap_top_drivers") or []
            if drivers:
                top_driver = drivers[0]
                parts.append(
                    f"Key model drivers indicate that {top_driver.get('feature_name', 'the leading feature')} "
                    f"is the strongest SHAP contributor for "
                    f"{self._metric_label(top_driver.get('target_name', explain.get('selected_target', 'the selected target')))}."
                )

        if "MovementAnalysisTool" in payload:
            movement = payload["MovementAnalysisTool"]
            summary_rows = movement.get("movement_summary") or []
            if summary_rows:
                primary = summary_rows[0]
                dominant_step = str(primary.get("dominant_step", "the leading step")).replace("_", " ")
                parts.append(
                    f"Management view: Movement analysis for {movement.get('latest_quarter', 'the latest quarter')} shows "
                    f"{self._metric_label(primary.get('metric', 'the selected metric'))} in {primary.get('product', 'the selected product')} / {primary.get('region', 'the selected region')} "
                    f"moved from {primary.get('opening_value', 'n/a')} to {primary.get('closing_value', 'n/a')}."
                )
                parts.append(
                    f"The dominant movement driver is {dominant_step}, and the top bridge steps are {primary.get('top_movement_steps', 'n/a')}."
                )

        if "NarrativeLookupTool" in payload:
            narrative = payload["NarrativeLookupTool"]
            sections = narrative.get("narrative_sections") or []
            if sections:
                parts.append(sections[0].get("statement_text", ""))

        if "FigureLookupTool" in payload:
            figures = payload["FigureLookupTool"].get("figures") or []
            if figures:
                parts.append(f"Relevant visuals include {figures[0].get('chart_title', 'the available reporting figure')}.")

        if "RunSummaryTool" in payload:
            run_summary = payload["RunSummaryTool"]
            completed = run_summary.get("completed_stages") or []
            if completed:
                parts.append(f"Completed workflow stages include {', '.join(completed[:5])}.")

        return " ".join(part for part in parts if part).strip()

    def _choose_connector(self, primary: str, secondary: str, *, management_mode: bool) -> str:
        primary_lower = primary.lower()
        secondary_lower = secondary.lower()
        if management_mode:
            if "forecast" in primary_lower or "forecast" in secondary_lower:
                return "The forward outlook suggests"
            if "validation" in primary_lower or "validation" in secondary_lower:
                return "Control metrics indicate"
            if "shap" in primary_lower or "feature" in secondary_lower:
                return "Key model drivers indicate"
            if "reserve" in primary_lower or "csm" in secondary_lower:
                return "Balance sheet indicators show"
            return "Supporting management evidence shows"
        if "forecast" in primary_lower or "forecast" in secondary_lower:
            return "The forecast evidence further suggests"
        if "validation" in primary_lower or "validation" in secondary_lower:
            return "Validation results also show"
        if "shap" in primary_lower or "feature" in secondary_lower:
            return "Explainability results indicate"
        return "Additional evidence shows"

    def _lowercase_first(self, text: str) -> str:
        if not text:
            return text
        return text[0].lower() + text[1:]

    def _metric_label(self, metric_name: str) -> str:
        labels = {
            "capital_ratio": "the synthetic capital-to-liability proxy ratio",
            "reserve_movement": "reserve movement",
            "csm_movement": "CSM movement",
        }
        return labels.get(metric_name, metric_name)


@dataclass(slots=True)
class OpenAILLMClient:
    """Optional OpenAI-backed client."""

    model: str = "gpt-4o-mini"

    def generate(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.model,
            input=prompt,
        )
        return response.output_text


@dataclass(slots=True)
class GeminiLLMClient:
    """Optional Gemini-backed client via the Gemini REST API."""

    model: str = "gemini-2.5-flash"
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"

    def generate(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

        endpoint = f"{self.api_base}/models/{self.model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ]
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Gemini API request failed: {message}") from exc

        body = json.loads(raw)
        candidates = body.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        return "\n".join(text_parts).strip()


@dataclass(slots=True)
class LocalLLMClient:
    """Wrapper around a local callable generator."""

    generator: Callable[[str], str]

    def generate(self, prompt: str) -> str:
        return self.generator(prompt)


def get_default_llm_client() -> LLMClient:
    """Return the default LLM client for local/offline operation."""

    provider = os.getenv("AIR_LLM_PROVIDER", "mock").lower()
    if provider == "openai":
        return OpenAILLMClient(model=os.getenv("AIR_OPENAI_MODEL", "gpt-4o-mini"))
    if provider == "gemini":
        return GeminiLLMClient(model=os.getenv("AIR_GEMINI_MODEL", "gemini-2.5-flash"))
    return MockLLMClient()

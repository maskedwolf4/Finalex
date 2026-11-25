"""Finalex's small LLM runtime.

This deliberately uses LiteLLM directly rather than the OpenAI Agents SDK.  Every
model invocation is recorded in MLflow when MLflow is configured; MLflow's local
file store is used by default so local development works without a tracking server.
"""

import json
import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from litellm import acompletion

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _model_name(model: Any) -> str:
    return getattr(model, "model", model) if model else os.getenv("FINALEX_MODEL", "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0")


def _mlflow():
    try:
        import mlflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "finalex"))
        return mlflow
    except Exception as error:  # Tracking must never prevent an analysis.
        logger.debug("MLflow is unavailable: %s", error)
        return None


@contextmanager
def trace(name: str):
    """Create a named MLflow run for an application operation."""
    mlflow = _mlflow()
    active = False
    if mlflow:
        try:
            mlflow.start_run(run_name=name, nested=mlflow.active_run() is not None)
            mlflow.set_tag("finalex.component", name)
            active = True
        except Exception as error:
            logger.debug("Could not start MLflow run: %s", error)
    try:
        yield
    finally:
        if active:
            try:
                mlflow.end_run()
            except Exception as error:
                logger.debug("Could not finish MLflow run: %s", error)


class LitellmModel:
    def __init__(self, model: str):
        self.model = model


class Agent(Generic[T]):
    def __init__(self, name: str, instructions: str, model: Any = None, tools=None, output_type=None, **_: Any):
        self.name, self.instructions, self.model = name, instructions, model
        self.tools, self.output_type = tools or [], output_type


@dataclass
class RunContextWrapper(Generic[T]):
    context: T


class RunHooks:
    pass


def function_tool(func):
    """Marker retained for ordinary Python callables; no SDK dependency is used."""
    return func


@dataclass
class RunResult:
    final_output: str

    def final_output_as(self, output_type):
        raw = self.final_output.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end >= start:
            raw = raw[start : end + 1]
        return output_type.model_validate_json(raw)


class Runner:
    @staticmethod
    async def run(agent: Agent, input: str, context=None, max_turns: int = 1, **_: Any) -> RunResult:
        """Run one direct LiteLLM completion and track request metadata in MLflow.

        The previous orchestration's side effects are explicit in the planner; this
        runtime intentionally does not hide autonomous tool execution.
        """
        model = _model_name(agent.model)
        system = agent.instructions
        if agent.output_type:
            system += "\nReturn only valid JSON matching this schema:\n" + json.dumps(agent.output_type.model_json_schema())
        mlflow = _mlflow()
        try:
            if mlflow:
                mlflow.log_params({"model": model, "agent_name": agent.name, "max_turns": max_turns})
                mlflow.log_text(input, "prompt.txt")
            response = await acompletion(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": input}])
            output = response.choices[0].message.content or ""
            if mlflow:
                mlflow.log_text(output, "response.txt")
            return RunResult(output)
        except Exception as error:
            if mlflow:
                try:
                    mlflow.set_tag("error", True)
                    mlflow.set_tag("error.message", str(error)[:500])
                except Exception:
                    pass
            raise

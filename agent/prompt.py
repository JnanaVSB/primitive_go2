"""Prompt builder.

Loads Jinja templates from the templates directory and renders them with
task-specific variables and trial history into the final prompt string
sent to the LLM.
"""

from pathlib import Path
from typing import Iterable
from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound


@dataclass
class TrialRecord:
    """One row of trial history, as seen by the prompt template.

    The template renders these into the history section. The runner builds
    them from the log of (code, reward, rationale) tuples.
    """
    iteration: int
    policy_summary: str
    reward: float | str
    rationale: str = ""


class PromptBuilder:
    """Renders task prompts from Jinja templates.

    Args:
        templates_dir: path to the templates directory.
    """

    def __init__(self, templates_dir: str | Path):
        self.templates_dir = Path(templates_dir)
        if not self.templates_dir.is_dir():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def build(
        self,
        task: str,
        iter_idx: int,
        max_iters: int,
        trial_history: Iterable[TrialRecord] = (),
        **kwargs,
    ) -> str:
        """Render the prompt for a task.

        Args:
            task:          template name without extension (e.g. 'code_policy').
            iter_idx:      1-based current iteration number.
            max_iters:     total iteration budget for this task.
            trial_history: prior (iteration, policy_summary, reward, rationale) records.
            **kwargs:      additional template variables (e.g. task_description,
                           distance_weight).

        Returns:
            The full prompt string to send to the LLM.
        """
        template_name = f"{task}.j2"
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            raise FileNotFoundError(
                f"Template '{template_name}' not found in {self.templates_dir}"
            )

        return template.render(
            iter_idx=iter_idx,
            max_iters=max_iters,
            trial_history=list(trial_history),
            **kwargs,
        )
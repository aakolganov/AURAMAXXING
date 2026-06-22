"""Config-file-driven surface generation.

Load a YAML config and run the full grow -> finalize -> saturate -> charge-correct ->
rules pipeline:

    from runner import run_from_config
    run_from_config("config.yaml")

or from the command line:

    python -m runner config.yaml          # run
    python -m runner --dry-run config.yaml  # validate + print the plan
"""
from .config import RunConfig, load_config
from .runner import run_from_config, resolve_plan, build_calculator

__all__ = ["RunConfig", "load_config", "run_from_config", "resolve_plan", "build_calculator"]

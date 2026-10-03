#!/usr/bin/env python3
"""Entry point for ROLL RLVR / StateTree GRPO training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root (`code/`) must be on PYTHONPATH before importing roll.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("PYTHONPATH", f"{_ROOT}:{os.environ.get('PYTHONPATH', '')}")


def _resolve_config_dir(config_path: str) -> str:
    """
    Resolve Hydra config directory.

    Accepts absolute paths, CWD-relative paths, or paths relative to examples/roll/.
    """
    p = Path(config_path)
    if p.is_absolute() and p.is_dir():
        return str(p)
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.is_dir():
        return str(cwd_candidate)
    here_candidate = (Path(__file__).resolve().parent / p).resolve()
    if here_candidate.is_dir():
        return str(here_candidate)
    raise FileNotFoundError(
        f"Config directory not found: {config_path} "
        f"(tried {cwd_candidate} and {here_candidate})"
    )


def main() -> None:
    from dacite import from_dict
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from roll.distributed.scheduler.initialize import init
    from roll.pipeline.rlvr.rlvr_config import RLVRConfig
    from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="statetree",
        help="Hydra config directory (absolute, CWD-relative, or relative to examples/roll/).",
    )
    parser.add_argument(
        "--config_name",
        default="grpo_qwen2.5_7b",
        help="Config file name without .yaml extension.",
    )
    args, overrides = parser.parse_known_args()

    config_dir = _resolve_config_dir(args.config_path)
    with initialize_config_dir(config_dir=config_dir, job_name="statetree_rlvr"):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config: RLVRConfig = from_dict(
        data_class=RLVRConfig,
        data=OmegaConf.to_container(cfg, resolve=True),
    )

    init()
    pipeline = RLVRPipeline(pipeline_config=ppo_config)
    pipeline.run()


if __name__ == "__main__":
    main()

"""Optional JSON config layer under argparse.

Every runner here is configured by plain CLI flags. `--config file.json` is a
thin convenience on top -- the JSON keys are the flags' names (dashes or
underscores), they become the parser's defaults, and any flag given explicitly
on the command line still wins. Unknown keys are an error (a typo must not
silently run the default). Keys starting with "_" (e.g. "_comment") are ignored.

Options that must be provided (by flag or by config) are declared WITHOUT
`required=True` and listed in `required=` instead, so a config can supply them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None,
                        help="JSON file whose keys are this script's flag names; command-line flags override it.")


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    required: Sequence[str] = (),
) -> argparse.Namespace:
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        cfg = json.loads(Path(pre.config).read_text())
        dests = {a.dest for a in parser._actions}
        normalized = {}
        for k, v in cfg.items():
            if k.startswith("_"):
                continue
            dest = k.replace("-", "_")
            if dest not in dests:
                parser.error(f"{pre.config}: unknown option {k!r} (valid: {sorted(d for d in dests if d != 'help')})")
            normalized[dest] = v
        parser.set_defaults(**normalized)
    args = parser.parse_args(argv)
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name, None) is None]
    if missing:
        parser.error(f"missing required option(s) (flag or --config): {', '.join(missing)}")
    return args

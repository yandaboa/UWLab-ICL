"""List all W&B projects for the current user/entity.

Usage:
    python list_wandb_projects.py                 # uses default entity from wandb login
    python list_wandb_projects.py --entity NAME   # override entity
"""
from __future__ import annotations

import argparse

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (username or team). Defaults to the logged-in user.",
    )
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    if entity is None:
        raise RuntimeError(
            "No entity provided and no default entity found. "
            "Run `wandb login` or pass --entity."
        )

    projects = list(api.projects(entity=entity))
    print(f"Entity: {entity}")
    print(f"Found {len(projects)} project(s):")
    for p in projects:
        print(f"  - {p.name}  (id={p.id}, created={p.created_at})")


if __name__ == "__main__":
    main()

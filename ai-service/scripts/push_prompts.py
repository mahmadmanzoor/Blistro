"""Upload the local prompts (app/agent/prompts.py) to Langfuse as chat prompts.

    uv run python -m scripts.push_prompts            # new version, label "staging"
    uv run python -m scripts.push_prompts --production

The service reads the `production` label, so pushing to staging changes nothing
until you promote that version in the Langfuse UI (or push with --production)."""

from __future__ import annotations

import argparse

from app.agent.prompts import ALL
from app.config import get_settings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--production", action="store_true")
    p.add_argument("--message", default="sync from app/agent/prompts.py")
    args = p.parse_args()

    s = get_settings()
    if not s.langfuse_enabled:
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY first.")
    from langfuse import Langfuse

    client = Langfuse(
        public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host
    )
    labels = ["production"] if args.production else ["staging"]
    for name, messages in ALL.items():
        prompt = client.create_prompt(
            name=name,
            type="chat",
            prompt=messages,
            labels=labels,
            commit_message=args.message,
        )
        print(f"{name}: v{prompt.version} {labels}")
    client.flush()


if __name__ == "__main__":
    main()

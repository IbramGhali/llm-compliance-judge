"""Generate the synthetic dataset without the CLI: `uv run python scripts/generate_transcripts.py`.

Same output as `uv run judge generate`. All data is synthetic and seeded.
"""

from __future__ import annotations

import argparse

from compliance_judge import synth
from compliance_judge.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    settings = Settings.from_env()
    calls = synth.generate(n=args.n, seed=args.seed)
    synth.write_dataset(calls, settings.data_dir)
    print(f"Wrote {len(calls)} transcripts to {settings.data_dir}")


if __name__ == "__main__":
    main()

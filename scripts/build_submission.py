#!/usr/bin/env python3
"""Build a hash-verified submission archive from a candidate manifest."""

import argparse
import json
from pathlib import Path

from keigo import Candidate
from keigo.submission import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("entrypoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    candidate = Candidate(**json.loads(args.candidate_manifest.read_text(encoding="utf-8")))
    result = build_bundle(candidate, args.entrypoint, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

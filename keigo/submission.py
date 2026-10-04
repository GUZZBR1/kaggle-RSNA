"""Reproducible, strategy-neutral bundle creation for a candidate entrypoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from .contracts import Candidate
from .identity import canonical_json, digest


def build_bundle(candidate: Candidate, source: str | Path, destination: str | Path) -> dict:
    """Bundle an already authored agent entrypoint and an identity manifest.

    This intentionally does not generate a policy. The source must expose the official
    Kaggle callback contract, which is checked by the selected environment adapter.
    """
    source_path, destination_path = Path(source), Path(destination)
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != candidate.artifact_sha256:
        raise ValueError("candidate artifact hash does not match submission entrypoint")
    manifest = {"schema_version": 1, "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.name,
                "entrypoint": candidate.entrypoint,
                "entrypoint_sha256": source_sha256}
    manifest["manifest_sha256"] = digest(manifest)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination_path.parent, suffix=".zip",
                                     delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, content in (("main.py", source_bytes),
                                  ("manifest.json", canonical_json(manifest) + "\n")):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                bundle.writestr(info, content)
        temporary_path.replace(destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {**manifest, "bundle_sha256": hashlib.sha256(destination_path.read_bytes()).hexdigest(),
            "path": str(destination_path)}

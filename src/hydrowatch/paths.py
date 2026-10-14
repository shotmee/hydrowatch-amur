"""Path resolution tolerant of upload-pipeline quirks.

Kaggle's web uploader, when you drag in top-level folders individually
(rather than zipping the whole dataset root), nests each one under itself:
``rasters/rasters/<event>/<aoi>/...``, ``vectors/vectors/...``, while files
dropped straight into the root (``pairs.csv``) stay unnested. The proper
fix is re-uploading the dataset as a single zip of the `hydrowatch_amur`
root — that preserves the original structure exactly. This module exists
so the pipeline still works on an already-published, oddly-nested copy
without needing to rewrite it in place (`/kaggle/input/...` is read-only
anyway).
"""
from __future__ import annotations

import glob
import os


def resolve(data_dir: str, rel_path: str) -> str:
    """Returns a real path under `data_dir` matching `rel_path`, trying the
    direct join first and falling back to a recursive search so extra
    nesting levels (single, double, or otherwise) don't matter.

    Raises FileNotFoundError with both the direct and searched location if
    nothing matches, so a genuinely missing file still fails loudly instead
    of silently resolving to the wrong thing.
    """
    direct = os.path.join(data_dir, rel_path)
    if os.path.exists(direct):
        return direct

    pattern = os.path.join(data_dir, "**", rel_path)
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"{rel_path!r} not found under {data_dir!r} (checked {direct!r} "
            f"and recursively under {data_dir!r})"
        )
    return matches[0]


def try_resolve(data_dir: str, rel_path: str) -> str | None:
    try:
        return resolve(data_dir, rel_path)
    except FileNotFoundError:
        return None

"""Frozen-app fallback when package dist-info is missing."""

from __future__ import annotations

import importlib.metadata as metadata

_orig_version = metadata.version


def _safe_version(distribution_name: str) -> str:
    try:
        return _orig_version(distribution_name)
    except metadata.PackageNotFoundError:
        return "0.0.0"


metadata.version = _safe_version

"""
fase7_predicoes_utils.py — Wrapper compatível para `pipeline.fase7.predicoes_utils`
"""
from __future__ import annotations

import sys

from fase7 import predicoes_utils as _impl

sys.modules[__name__] = _impl

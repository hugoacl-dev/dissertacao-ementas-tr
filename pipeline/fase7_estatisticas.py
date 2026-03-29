"""
fase7_estatisticas.py — Wrapper compatível para `pipeline.fase7.estatisticas`
"""
from __future__ import annotations

import sys

from fase7 import estatisticas as _impl

sys.modules[__name__] = _impl

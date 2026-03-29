"""
fase7_metricas.py — Wrapper compatível para `pipeline.fase7.metricas`
"""
from __future__ import annotations

import sys

from fase7 import metricas as _impl

sys.modules[__name__] = _impl

"""
fase7_casos.py — Wrapper compatível para `pipeline.fase7.casos_avaliacao`
"""
from __future__ import annotations

import sys

from fase7 import casos_avaliacao as _impl

sys.modules[__name__] = _impl

"""
fase7_protocolo.py — Wrapper compatível para `pipeline.fase7.protocolo`
"""
from __future__ import annotations

import sys

from fase7 import protocolo as _impl

sys.modules[__name__] = _impl

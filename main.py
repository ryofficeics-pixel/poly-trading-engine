"""
main.py — FastAPI entrypoint shim
===================================
Some deployment platforms (Vercel, some Railway configs) scan for
main.py or main:app by default. This re-exports the app from ws_server.py
so any platform finds a valid entrypoint regardless of how it scans.

Entrypoint references (all equivalent):
  ws_server:app   ← primary (declared in pyproject.toml + Procfile)
  main:app        ← this file (fallback for platforms that scan main.py)
"""

from ws_server import app  # noqa: F401 — re-export for platform discovery

__all__ = ["app"]

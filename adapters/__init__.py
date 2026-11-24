"""Adapters over the public Memnoth surfaces only.

This package never imports a private crate and never names a private repository
path. It talks to a binary over its published JSON envelope or its newline
delimited JSON-RPC stdio wire.
"""

__all__ = ["judge", "mcp_stdio", "mn_json"]

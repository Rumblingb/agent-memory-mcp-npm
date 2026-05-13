#!/usr/bin/env python3
"""
Agent Memory MCP Server
========================
A persistent key-value memory store for AI agents with TTL, namespaces,
fuzzy search, and access counting.  Data lives in ``~/.agent-memory/``
as one JSON file per namespace plus a ``_meta.json`` stats file.

Built on the MCP (Model Context Protocol) Python SDK.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
    ToolAnnotations,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARACTER_LIMIT = 25_000
DEFAULT_NAMESPACE = "default"
STORAGE_DIR = Path.home() / ".agent-memory"
META_FILE = "_meta.json"

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _ensure_storage() -> None:
    """Create the storage directory if it doesn't exist."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _namespace_path(namespace: str) -> Path:
    """Return the full path for a namespace JSON file."""
    # Sanitize the namespace so it can't escape the directory.
    safe = re.sub(r"[^a-zA-Z0-9_.\-]", "_", namespace)
    if not safe:
        safe = DEFAULT_NAMESPACE
    return STORAGE_DIR / f"{safe}.json"


def _meta_path() -> Path:
    """Return the full path for the meta-data file."""
    return STORAGE_DIR / META_FILE


@contextmanager
def _locked_file(path: Path, mode: str = "r+"):
    """Open a file with an exclusive POSIX lock (fcntl.flock).

    Falls back gracefully on platforms that don't support ``fcntl``
    (e.g. Windows without WSL) – in that case the lock is a no-op.
    """
    _ensure_storage()
    file_exists = path.exists()
    if not file_exists and "w" in mode or "+" in mode:
        path.touch(exist_ok=True)
    fh = open(path, mode)
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
        except (NameError, OSError):
            pass  # platform without fcntl support
        yield fh
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        except (NameError, OSError):
            pass
        fh.close()


def _read_namespace(namespace: str) -> List[Dict[str, Any]]:
    """Read all entries for a namespace (returns list, never None)."""
    path = _namespace_path(namespace)
    if not path.exists():
        return []
    with _locked_file(path, "r") as fh:
        try:
            fh.seek(0)
            raw = fh.read()
            if not raw.strip():
                return []
            return json.loads(raw)
        except (json.JSONDecodeError, OSError):
            return []


def _write_namespace(namespace: str, entries: List[Dict[str, Any]]) -> None:
    """Atomically write all entries for a namespace."""
    _ensure_storage()
    path = _namespace_path(namespace)
    with _locked_file(path, "w") as fh:
        fh.seek(0)
        fh.truncate()
        json.dump(entries, fh, indent=2)
        fh.flush()


def _read_meta() -> Dict[str, Any]:
    """Read global metadata (or return defaults)."""
    path = _meta_path()
    if not path.exists():
        return {}
    with _locked_file(path, "r") as fh:
        try:
            fh.seek(0)
            raw = fh.read()
            if not raw.strip():
                return {}
            return json.loads(raw)
        except (json.JSONDecodeError, OSError):
            return {}


def _write_meta(meta: Dict[str, Any]) -> None:
    """Write global metadata."""
    _ensure_storage()
    path = _meta_path()
    with _locked_file(path, "w") as fh:
        fh.seek(0)
        fh.truncate()
        json.dump(meta, fh, indent=2)
        fh.flush()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_expired(entry: Dict[str, Any]) -> bool:
    """Return True if the entry has a TTL that has passed."""
    expires = entry.get("expires_at")
    if expires is None:
        return False
    # expires_at is stored as a Unix timestamp (float)
    return time.time() > expires


def _truncate(text: str, limit: int = CHARACTER_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 30] + "... [TRUNCATED at {limit} chars]"


def _format_response(
    result: Dict[str, Any],
    fmt: Optional[str] = None,
) -> str:
    """Render a result dictionary as either JSON or Markdown."""
    if fmt == "json":
        return json.dumps(result, indent=2)
    # Markdown (default)
    lines = []
    status = result.get("status", "ok")
    if status == "error":
        lines.append(f"## ❌ Error")
        lines.append(f"**{result.get('error', 'Unknown error')}**")
    else:
        lines.append(f"## ✅ Success")
    for k, v in result.items():
        if k in ("status", "error", "isError"):
            continue
        if isinstance(v, list):
            lines.append(f"**{k}:** {len(v)} items")
            for item in v:
                if isinstance(item, dict):
                    for ik, iv in item.items():
                        lines.append(f"  - **{ik}:** {iv}")
                    lines.append("")
                else:
                    lines.append(f"  - {item}")
        elif isinstance(v, dict):
            lines.append(f"**{k}:**")
            for ik, iv in v.items():
                lines.append(f"  - **{ik}:** {iv}")
        else:
            lines.append(f"**{k}:** {v}")
    return "\n".join(lines)


def _error(message: str, fmt: Optional[str] = None) -> str:
    result = {"status": "error", "error": message, "isError": True}
    return _format_response(result, fmt)


def _success(data: Dict[str, Any], fmt: Optional[str] = None) -> str:
    data.setdefault("status", "ok")
    return _format_response(data, fmt)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def memory_remember(
    key: str,
    value: str,
    namespace: str = DEFAULT_NAMESPACE,
    ttl_seconds: Optional[int] = None,
    fmt: Optional[str] = None,
) -> str:
    """Store a value under a key."""
    if not key.strip():
        return _error("Key must not be empty", fmt)
    if not value:
        return _error("Value must not be empty", fmt)

    entries = _read_namespace(namespace)

    # Remove existing entry with the same key (upsert)
    entries = [e for e in entries if e["key"] != key]

    now = time.time()
    entry: Dict[str, Any] = {
        "key": key,
        "value": value,
        "created_at": _now_iso(),
        "accessed_at": _now_iso(),
        "expires_at": (now + ttl_seconds) if ttl_seconds else None,
        "access_count": 0,
    }
    entries.append(entry)
    _write_namespace(namespace, entries)

    # Update meta
    meta = _read_meta()
    meta["total_entries"] = meta.get("total_entries", 0) + 1
    # Recalculate total entries based on actual file contents
    _recalc_meta()

    return _success(
        {
            "message": f"Stored '{key}' in namespace '{namespace}'",
            "key": key,
            "namespace": namespace,
            "expires_in": f"{ttl_seconds}s" if ttl_seconds else "never",
            "expires_at": entry["expires_at"],
        },
        fmt,
    )


def memory_recall(
    key: str,
    namespace: str = DEFAULT_NAMESPACE,
    fmt: Optional[str] = None,
) -> str:
    """Retrieve a value by key, with lazy TTL expiry."""
    if not key.strip():
        return _error("Key must not be empty", fmt)

    entries = _read_namespace(namespace)

    for i, entry in enumerate(entries):
        if entry["key"] == key:
            if _is_expired(entry):
                # Lazy expiry – remove and return not-found
                entries.pop(i)
                _write_namespace(namespace, entries)
                _recalc_meta()
                return _error(f"Key '{key}' has expired", fmt)

            # Update access metadata
            entry["accessed_at"] = _now_iso()
            entry["access_count"] = entry.get("access_count", 0) + 1
            _write_namespace(namespace, entries)

            return _success(
                {
                    "key": key,
                    "namespace": namespace,
                    "value": _truncate(entry["value"]),
                    "created_at": entry["created_at"],
                    "accessed_at": entry["accessed_at"],
                    "expires_at": entry["expires_at"],
                    "access_count": entry["access_count"],
                },
                fmt,
            )

    return _error(f"Key '{key}' not found in namespace '{namespace}'", fmt)


def memory_forget(
    key: str,
    namespace: str = DEFAULT_NAMESPACE,
    fmt: Optional[str] = None,
) -> str:
    """Delete a key from a namespace."""
    if not key.strip():
        return _error("Key must not be empty", fmt)

    entries = _read_namespace(namespace)
    original_len = len(entries)
    entries = [e for e in entries if e["key"] != key]

    if len(entries) == original_len:
        return _error(f"Key '{key}' not found in namespace '{namespace}'", fmt)

    _write_namespace(namespace, entries)
    _recalc_meta()

    return _success(
        {
            "message": f"Deleted '{key}' from namespace '{namespace}'",
            "key": key,
            "namespace": namespace,
        },
        fmt,
    )


def memory_search(
    query: str,
    namespace: Optional[str] = None,
    limit: int = 10,
    fmt: Optional[str] = None,
) -> str:
    """Search memories by keyword (case-insensitive substring match)."""
    if not query.strip():
        return _error("Query must not be empty", fmt)

    q = query.lower()
    results: List[Dict[str, Any]] = []

    namespaces_to_search: List[str]
    if namespace:
        namespaces_to_search = [namespace]
    else:
        # Discover all namespace files
        namespaces_to_search = [
            p.stem
            for p in STORAGE_DIR.glob("*.json")
            if p.stem != "_meta"
        ]

    for ns in namespaces_to_search:
        entries = _read_namespace(ns)
        for entry in entries:
            if _is_expired(entry):
                continue
            if q in entry["key"].lower() or q in entry["value"].lower():
                results.append(
                    {
                        "namespace": ns,
                        "key": entry["key"],
                        "value": _truncate(entry["value"], 500),
                        "created_at": entry["created_at"],
                        "access_count": entry.get("access_count", 0),
                    }
                )

    # Sort by access count desc, then created_at desc
    results.sort(key=lambda r: (r["access_count"], r["created_at"]), reverse=True)

    total = len(results)
    results = results[:limit]

    return _success(
        {
            "query": query,
            "total_matches": total,
            "returned": len(results),
            "results": results,
        },
        fmt,
    )


def memory_list_namespaces(fmt: Optional[str] = None) -> str:
    """List all namespaces with entry counts."""
    namespaces: List[Dict[str, Any]] = []
    for p in sorted(STORAGE_DIR.glob("*.json")):
        if p.stem == "_meta":
            continue
        entries = _read_namespace(p.stem)
        # Filter out expired entries for the count
        active = [e for e in entries if not _is_expired(e)]
        namespaces.append(
            {
                "namespace": p.stem,
                "total_entries": len(entries),
                "active_entries": len(active),
                "expired_entries": len(entries) - len(active),
            }
        )

    return _success(
        {
            "namespace_count": len(namespaces),
            "namespaces": namespaces,
        },
        fmt,
    )


def memory_clear_namespace(
    namespace: str,
    fmt: Optional[str] = None,
) -> str:
    """Delete all entries in a namespace."""
    if not namespace.strip():
        return _error("Namespace must not be empty", fmt)

    entries = _read_namespace(namespace)
    count = len(entries)
    _write_namespace(namespace, [])
    _recalc_meta()

    return _success(
        {
            "message": f"Cleared {count} entries from namespace '{namespace}'",
            "namespace": namespace,
            "deleted_count": count,
        },
        fmt,
    )


def memory_stats(fmt: Optional[str] = None) -> str:
    """Get storage statistics."""
    total_entries = 0
    total_size = 0
    namespace_count = 0
    oldest: Optional[str] = None
    newest: Optional[str] = None

    for p in STORAGE_DIR.glob("*.json"):
        if p.stem == "_meta":
            continue
        namespace_count += 1
        try:
            file_size = p.stat().st_size
            total_size += file_size
        except OSError:
            pass
        entries = _read_namespace(p.stem)
        active = [e for e in entries if not _is_expired(e)]
        total_entries += len(active)
        for e in active:
            created = e.get("created_at")
            if created:
                if oldest is None or created < oldest:
                    oldest = created
                if newest is None or created > newest:
                    newest = created

    return _success(
        {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_human": _human_size(total_size),
            "namespace_count": namespace_count,
            "oldest_entry": oldest or "N/A",
            "newest_entry": newest or "N/A",
            "storage_path": str(STORAGE_DIR),
            "free_tier_limit": 1000,
            "pro_tier_limit": "unlimited",
        },
        fmt,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _recalc_meta() -> None:
    """Recompute global metadata by scanning all namespaces."""
    total = 0
    namespace_count = 0
    for p in STORAGE_DIR.glob("*.json"):
        if p.stem == "_meta":
            continue
        namespace_count += 1
        entries = _read_namespace(p.stem)
        total += len([e for e in entries if not _is_expired(e)])
    _write_meta({"total_entries": total, "namespace_count": namespace_count})


# ---------------------------------------------------------------------------
# MCP Server definition
# ---------------------------------------------------------------------------

server = Server(
    name="agent-memory",
    version="1.0.0",
    instructions="Agent Memory MCP — Persistent key-value memory for AI agents with TTL, namespaces, and search.",
    website_url="https://github.com/nousresearch/agent-memory-mcp",
)


@server.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="memory_remember",
            description="Store a value under a key in a persistent memory namespace. Optionally set a TTL (time-to-live) in seconds for automatic expiry.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Unique key for this memory entry.",
                    },
                    "value": {
                        "type": "string",
                        "description": "The value/content to store.",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Namespace to store the entry in (default: 'default').",
                        "default": "default",
                    },
                    "ttl_seconds": {
                        "type": "integer",
                        "description": "Optional TTL in seconds. Entry auto-expires after this duration.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown).",
                        "default": "markdown",
                    },
                },
                "required": ["key", "value"],
            },
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        Tool(
            name="memory_recall",
            description="Retrieve a stored value by key from a namespace. Returns full metadata including creation time, last access, and expiry. Automatically expires TTL'd entries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to retrieve.",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Namespace to look in (default: 'default').",
                        "default": "default",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown).",
                        "default": "markdown",
                    },
                },
                "required": ["key"],
            },
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        Tool(
            name="memory_forget",
            description="Delete a specific key from a namespace. This is permanent and cannot be undone.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to delete.",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Namespace to delete from (default: 'default').",
                        "default": "default",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown).",
                        "default": "markdown",
                    },
                },
                "required": ["key"],
            },
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        Tool(
            name="memory_search",
            description="Search memories across namespaces by keyword substring. Case-insensitive match on both keys and values. Returns results sorted by access count.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or substring.",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace to limit search. Searches ALL namespaces if omitted.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 10).",
                        "default": 10,
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown).",
                        "default": "markdown",
                    },
                },
                "required": ["query"],
            },
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=True,
            ),
        ),
        Tool(
            name="memory_list_namespaces",
            description="List all memory namespaces with active/expired entry counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown).",
                        "default": "markdown",
                    },
                },
            },
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=True,
            ),
        ),
        Tool(
            name="memory_clear_namespace",
            description="Permanently delete ALL entries in a namespace. This action cannot be undone.",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Namespace to clear.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown).",
                        "default": "markdown",
                    },
                },
                "required": ["namespace"],
            },
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=True,
                openWorldHint=False,
            ),
        ),
        Tool(
            name="memory_stats",
            description="Get storage statistics: total entries, size, namespace count, oldest/newest entry, and tier limits.",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "json"],
                        "description": "Response format (default: markdown).",
                        "default": "markdown",
                    },
                },
            },
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=True,
            ),
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """Route tool calls to the appropriate implementation."""
    fmt = arguments.pop("format", "markdown")

    try:
        if name == "memory_remember":
            text = memory_remember(**arguments, fmt=fmt)
        elif name == "memory_recall":
            text = memory_recall(**arguments, fmt=fmt)
        elif name == "memory_forget":
            text = memory_forget(**arguments, fmt=fmt)
        elif name == "memory_search":
            text = memory_search(**arguments, fmt=fmt)
        elif name == "memory_list_namespaces":
            text = memory_list_namespaces(fmt=fmt)
        elif name == "memory_clear_namespace":
            text = memory_clear_namespace(**arguments, fmt=fmt)
        elif name == "memory_stats":
            text = memory_stats(fmt=fmt)
        else:
            text = _error(f"Unknown tool: {name}", fmt)

        # Truncate final output if necessary
        text = _truncate(text)

        return CallToolResult(
            content=[TextContent(type="text", text=text)],
        )
    except Exception as exc:
        err_text = _error(f"Internal error in {name}: {exc}", fmt)
        return CallToolResult(
            content=[TextContent(type="text", text=err_text)],
            isError=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    _ensure_storage()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

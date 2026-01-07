"""
File name: mcp_tracking.py
Author: Luigi Saetta
Date last modified: 2026-01-07
Python Version: 3.11

Description:
    MCP (Model Context Protocol) server for client tracking using local Markdown files.
    It exposes tools that allow an LLM assistant to:
      - Create a new tracking folder and MD file for a client
      - Read existing tracking information (raw Markdown, optionally tail-limited)
      - Append new tracking updates to the client's Markdown file
      - List clients present under the tracking root folder

Security:
    Uses shared MCP utilities in mcp_utils.py.
    If ENABLE_JWT_TOKEN=True in config.py, requests must include a valid JWT token
    verifiable via OCI IAM JWKS.

Transport:
    Uses TRANSPORT from config.py, recommended "streamable-http".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import logging

from oci_models import get_llm
from mcp_utils import create_server, run_server

from config import DEBUG, DEFAULT_TRACKING_ROOT

logger = logging.getLogger("mcp_tracking")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

mcp = create_server("Tracking MCP")

# -----------------------
# Defaults / constraints
# -----------------------
DEFAULT_FILENAME = "tracking.md"

MAX_FILE_CHARS_HARD = 250_000  # hard cap for read-back (safety)
DEFAULT_TAIL_CHARS = 50_000  # default tail for tool responses


# -----------------------
# Helper utilities
# -----------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_client_id(client_id: str) -> str:
    """
    Very small hardening against path traversal and weird names.
    Keep it conservative: letters, digits, '-', '_', '.' only.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(ch for ch in (client_id or "").strip() if ch in allowed)
    if not cleaned:
        raise ValueError("Invalid client_id (empty or unsupported characters).")
    return cleaned


def _base_dir(tracking_root: str) -> Path:
    return Path(tracking_root).expanduser().resolve()


def _client_dir(tracking_root: str, client_id: str) -> Path:
    base = _base_dir(tracking_root)
    cid = _safe_client_id(client_id)
    p = (base / cid).resolve()

    # Ensure p is inside base (anti traversal)
    if base != p and base not in p.parents:
        raise ValueError("Invalid client path.")
    return p


def _client_file(tracking_root: str, client_id: str, filename: str) -> Path:
    cdir = _client_dir(tracking_root, client_id)
    f = (cdir / filename).resolve()

    # Ensure file is inside client dir (anti traversal)
    if cdir != f.parent and cdir not in f.parents:
        raise ValueError("Invalid file path.")
    return f


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_text_tail(p: Path, tail_chars: int) -> str:
    """
    Reads a file and returns the last tail_chars characters (bounded).
    """
    txt = p.read_text(encoding="utf-8")
    if len(txt) > MAX_FILE_CHARS_HARD:
        # If file exploded, return only tail from the hard cap window
        txt = txt[-MAX_FILE_CHARS_HARD:]
    if tail_chars <= 0:
        return ""
    if len(txt) <= tail_chars:
        return txt
    return txt[-tail_chars:]


def _default_md_template(client_id: str) -> str:
    now = _utc_now_iso()
    return (
        f"# Tracking — {client_id}\n\n"
        f"- Created: {now}\n"
        f"- Last update: {now}\n\n"
        "## Summary\n\n"
        "_(Contesto, obiettivi, stakeholder, stato sintetico.)_\n\n"
        "## History\n\n"
        f"### {now}\n"
        "- Initial file created.\n"
    )


def _update_last_update_line(md: str, ts: str) -> str:
    marker = "- Last update:"
    if marker not in md:
        return md
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(marker):
            lines[i] = f"{marker} {ts}"
            return "\n".join(lines)
    return md


def _append_history(md_existing: str, update_md: str, ts: str) -> str:
    update_md = (update_md or "").strip()
    if not update_md:
        raise ValueError("update_md is empty.")

    md_existing = _update_last_update_line(md_existing, ts)

    block = f"\n\n### {ts}\n{update_md}\n"

    if "## History" in md_existing:
        return md_existing + block

    # If missing History section, add it at the end
    return md_existing + "\n\n## History" + block


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Calls the LLM with the given system and user prompts.
    Returns the LLM response content as string.
    Raises RuntimeError on failure.

    Args:
        system_prompt: system-level instructions
        user_prompt: user-level prompt
    """
    llm = get_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = llm.invoke(messages)

        if DEBUG:
            logger.info("LLM response received.")
            logger.info("LLM response content: %r", response.content)

        return response.content
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Error in calling LLM. EXception: {e}") from e


# -----------------------
# MCP tools
# -----------------------


@mcp.tool()
def tracking_init_client(
    client_id: str,
    tracking_root: str = DEFAULT_TRACKING_ROOT,
    filename: str = DEFAULT_FILENAME,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Create a new client folder and Markdown tracking file using a standard template.

    Returns:
        {"client_id": str, "path": str, "created": bool, "message": str}
    """
    logger.info(
        "tracking_init_client client_id=%r root=%r filename=%r overwrite=%r",
        client_id,
        tracking_root,
        filename,
        overwrite,
    )

    try:
        cid = _safe_client_id(client_id)
        md_path = _client_file(tracking_root, cid, filename)
        _ensure_dir(md_path.parent)

        if md_path.exists() and not overwrite:
            return {
                "client_id": cid,
                "path": str(md_path),
                "created": False,
                "message": "File already exists (set overwrite=true to replace).",
            }

        md_path.write_text(_default_md_template(cid), encoding="utf-8")
        return {
            "client_id": cid,
            "path": str(md_path),
            "created": True,
            "message": "Created.",
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Error in tracking_init_client: %r", e)
        raise RuntimeError(f"Error creating client tracking file: {e}") from e


@mcp.tool()
def tracking_read(
    client_id: str,
    tracking_root: str = DEFAULT_TRACKING_ROOT,
    filename: str = DEFAULT_FILENAME,
    tail_chars: int = DEFAULT_TAIL_CHARS,
) -> Dict[str, Any]:
    """
    Read the client's tracking Markdown.

    Notes:
        Returns only the last tail_chars characters by default to keep context bounded.

    Returns:
        {"client_id": str, "path": str, "markdown": str}
    """
    logger.info(
        "tracking_read client_id=%r root=%r filename=%r tail_chars=%r",
        client_id,
        tracking_root,
        filename,
        tail_chars,
    )

    try:
        cid = _safe_client_id(client_id)
        md_path = _client_file(tracking_root, cid, filename)
        if not md_path.exists():
            raise FileNotFoundError(f"Tracking file not found: {md_path}")

        txt = _read_text_tail(md_path, int(tail_chars))
        return {"client_id": cid, "path": str(md_path), "markdown": txt}
    except Exception as e:  # noqa: BLE001
        logger.error("Error in tracking_read: %r", e)
        raise RuntimeError(f"Error reading tracking file: {e}") from e


@mcp.tool()
def tracking_update(
    client_id: str,
    update_md: str,
    tracking_root: str = DEFAULT_TRACKING_ROOT,
    filename: str = DEFAULT_FILENAME,
    create_if_missing: bool = True,
    return_tail_chars: int = DEFAULT_TAIL_CHARS,
) -> Dict[str, Any]:
    """
    Append new tracking info as a timestamped entry under '## History'.

    Args:
        update_md: markdown to append (e.g., "- Done ...", "Decision: ...", etc.)
        create_if_missing: if True, creates the file from template if missing
        return_tail_chars: tail chars to return after update (for confirmation)

    Returns:
        {
          "client_id": str,
          "path": str,
          "timestamp": str,
          "updated": bool,
          "markdown_tail": str
        }
    """
    logger.info(
        "tracking_update client_id=%r root=%r filename=%r create_if_missing=%r",
        client_id,
        tracking_root,
        filename,
        create_if_missing,
    )

    try:
        cid = _safe_client_id(client_id)
        md_path = _client_file(tracking_root, cid, filename)
        _ensure_dir(md_path.parent)

        if not md_path.exists():
            if not create_if_missing:
                raise FileNotFoundError(f"Tracking file not found: {md_path}")
            md_path.write_text(_default_md_template(cid), encoding="utf-8")

        existing = md_path.read_text(encoding="utf-8")
        ts = _utc_now_iso()
        updated = _append_history(existing, update_md, ts)

        md_path.write_text(updated, encoding="utf-8")
        tail = _read_text_tail(md_path, int(return_tail_chars))

        return {
            "client_id": cid,
            "path": str(md_path),
            "timestamp": ts,
            "updated": True,
            "markdown_tail": tail,
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Error in tracking_update: %r", e)
        raise RuntimeError(f"Error updating tracking file: {e}") from e


@mcp.tool()
def tracking_list_clients(tracking_root: str = DEFAULT_TRACKING_ROOT) -> Dict[str, Any]:
    """
    List all the customer/client IDs for which we have tracking info.

    Returns:
        {"count": int, "clients": [str]}
    """
    logger.info("tracking_list_clients root=%r", tracking_root)

    try:
        base = _base_dir(tracking_root)
        if not base.exists():
            return {"count": 0, "clients": []}

        clients: List[str] = []
        for p in sorted(base.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                clients.append(p.name)

        return {"count": len(clients), "clients": clients}
    except Exception as e:  # noqa: BLE001
        logger.error("Error in tracking_list_clients: %r", e)
        raise RuntimeError(f"Error listing clients: {e}") from e


@mcp.tool()
def tracking_propose_update(
    client_id: str,
    proposed_update: str,
    tracking_root: str = DEFAULT_TRACKING_ROOT,
    filename: str = DEFAULT_FILENAME,
    history_tail_chars: int = 80_000,
) -> Dict[str, Any]:
    """
    Takes a proposed update (prompt) and returns:
      1) a contradiction check against past history (best-effort)
      2) a rewritten/corrected update consistent with the history

    IMPORTANT:
      - This tool does NOT write to disk.
      - It is meant to be followed by tracking_update() if accepted.

    Returns:
      {
        "client_id": str,
        "contradictions": {"has_contradictions": bool, "items": [...]},
        "rewritten_update_md": str,
        "notes": str
      }
    """
    logger.info(
        "tracking_propose_update client_id=%r root=%r filename=%r tail=%r",
        client_id,
        tracking_root,
        filename,
        history_tail_chars,
    )

    try:
        cid = _safe_client_id(client_id)
        md_path = _client_file(tracking_root, cid, filename)

        if not md_path.exists():
            raise FileNotFoundError(
                f"Tracking file not found: {md_path}. Create it first with tracking_init_client()."
            )

        history_md = _read_text_tail(md_path, int(history_tail_chars))

        system_prompt = (
            "You are a careful enterprise note-taker. "
            "Your job: validate a proposed tracking update against the existing history, "
            "spot obvious contradictions, and rewrite the update to be consistent, precise, "
            "and useful.\n\n"
            "Rules:\n"
            "- Only flag contradictions that are evident from the provided history.\n"
            "- If it's unclear, mark as 'uncertain' rather than claiming contradiction.\n"
            "- Rewrite in concise Markdown.\n"
            "- Preserve facts in the proposal unless they contradict the history.\n"
            "- If the proposal introduces new facts that conflict, suggest a safer wording.\n"
            "- Output MUST be valid JSON with the required schema and nothing else.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "contradictions": {\n'
            '    "has_contradictions": boolean,\n'
            '    "items": [\n'
            "      {\n"
            '        "claim": string,\n'
            '        "conflicts_with": string,\n'
            '        "severity": "high"|"medium"|"low"|"uncertain",\n'
            '        "reason": string\n'
            "      }\n"
            "    ]\n"
            "  },\n"
            '  "rewritten_update_md": string,\n'
            '  "notes": string\n'
            "}\n"
        )

        user_prompt = (
            "EXISTING HISTORY (tail):\n"
            "-----\n"
            f"{history_md}\n"
            "-----\n\n"
            "PROPOSED UPDATE:\n"
            "-----\n"
            f"{proposed_update}\n"
            "-----\n"
        )

        raw = _call_llm(system_prompt=system_prompt, user_prompt=user_prompt)

        try:
            obj = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"LLM did not return valid JSON. Error={e}. Raw={raw}"
            ) from e

        # Minimal shape checks (fail fast if wildly off)
        if "contradictions" not in obj or "rewritten_update_md" not in obj:
            raise RuntimeError(f"LLM JSON missing required keys. Raw={obj}")

        return {
            "client_id": cid,
            "contradictions": obj.get("contradictions", {}),
            "rewritten_update_md": (obj.get("rewritten_update_md") or "").strip(),
            "notes": (obj.get("notes") or "").strip(),
        }

    except Exception as e:  # noqa: BLE001
        logger.error("Error in tracking_propose_update: %r", e)
        raise RuntimeError(f"Error proposing update: {e}") from e


@mcp.tool()
def tracking_summarize(
    client_id: str,
    tracking_root: str = DEFAULT_TRACKING_ROOT,
    filename: str = DEFAULT_FILENAME,
    history_tail_chars: int = 200_000,
) -> Dict[str, Any]:
    """
    Summarize the client's tracking history into the most relevant information
    (NOT a chronology). Focus on:
      - Current status
      - Goals / scope
      - Key decisions and rationale
      - Open risks/issues
      - Next steps / owners (if available)
      - Important constraints (security, compliance, timelines)

    Returns:
      {
        "client_id": str,
        "summary_md": str
      }
    """
    logger.info(
        "tracking_summarize client_id=%r root=%r filename=%r tail=%r",
        client_id,
        tracking_root,
        filename,
        history_tail_chars,
    )

    try:
        cid = _safe_client_id(client_id)
        md_path = _client_file(tracking_root, cid, filename)

        if not md_path.exists():
            raise FileNotFoundError(
                f"Tracking file not found: {md_path}. Create it first with tracking_init_client()."
            )

        history_md = _read_text_tail(md_path, int(history_tail_chars))

        system_prompt = (
            "You are an expert enterprise analyst. "
            "Your task is to summarize the provided tracking history into ONLY the most relevant "
            "information, avoiding chronology and avoiding repeating daily logs.\n\n"
            "Rules:\n"
            "- Do NOT write a timeline.\n"
            "- Prefer aggregated insights: status, goals, decisions, risks, blockers, next steps.\n"
            "- If information is missing, say 'Unknown' rather than inventing.\n"
            "- Keep it concise but complete.\n"
            "- Output MUST be valid JSON with the required schema and nothing else.\n\n"
            "Required JSON schema:\n"
            '{ "summary_md": string }\n'
        )

        user_prompt = "TRACKING HISTORY (tail):\n" "-----\n" f"{history_md}\n" "-----\n"

        raw = _call_llm(system_prompt=system_prompt, user_prompt=user_prompt)

        try:
            obj = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"LLM did not return valid JSON. Error={e}. Raw={raw}"
            ) from e

        summary_md = (obj.get("summary_md") or "").strip()
        if not summary_md:
            raise RuntimeError(f"LLM returned empty summary_md. Raw={obj}")

        return {"client_id": cid, "summary_md": summary_md}

    except Exception as e:  # noqa: BLE001
        logger.error("Error in tracking_summarize: %r", e)
        raise RuntimeError(f"Error summarizing tracking: {e}") from e


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    run_server(mcp)

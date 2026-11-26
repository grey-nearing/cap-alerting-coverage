"""Parameterized database connectors (SQLite/PostgreSQL via SQLAlchemy, optional Cloud Spanner)."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

TIMESTAMP_COLUMNS: frozenset[str] = frozenset(
    {
        "sent_at",
        "created_at",
        "updated_at",
        "start_date",
        "end_date",
    }
)


def sanitize_val(val: Any, col_name: str | None = None, max_bytes: int = 2_500_000) -> Any:
    """Normalize Python values safely for relational or Spanner storage.

    ISO-8601 strings are parsed to UTC ``datetime`` objects *only* for explicit timestamp columns,
    leaving identifier, headline, and description fields untouched as strings.
    """
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.replace(tzinfo=timezone.utc) if val.tzinfo is None else val
    if isinstance(val, date):
        return val
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    if isinstance(val, bool):
        return bool(val)
    if isinstance(val, str):
        if col_name in TIMESTAMP_COLUMNS and 10 <= len(val) < 64:
            try:
                from dateutil import parser as dt_parser

                dt = dt_parser.parse(val)
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
            except Exception:
                pass
        if len(val) > 600_000:
            encoded = val.encode("utf-8", errors="ignore")
            if len(encoded) > max_bytes:
                return encoded[:max_bytes].decode("utf-8", errors="ignore")
        return val
    if isinstance(val, (int, float)):
        return val
    text_val = str(val)
    if len(text_val) > 600_000:
        encoded = text_val.encode("utf-8", errors="ignore")
        if len(encoded) > max_bytes:
            return encoded[:max_bytes].decode("utf-8", errors="ignore")
    return text_val


def get_sqlalchemy_engine(db_url: str | None = None):
    """Create a SQLAlchemy engine from ``db_url`` or ``CAP_GAP_DB_URL`` (supports SQLite & PostgreSQL)."""
    try:
        import sqlalchemy
    except ModuleNotFoundError as exc:
        raise ImportError(
            "SQLAlchemy is required for Tier-2 database workflows. Install with: pip install -e '.[tier2]'"
        ) from exc

    url = db_url or os.environ.get("CAP_GAP_DB_URL", "").strip()
    if not url:
        raise RuntimeError(
            "Set CAP_GAP_DB_URL (e.g., 'sqlite:///cap_alerts.db' or "
            "'postgresql+psycopg2://user:pass@host:5432/dbname') to connect."
        )
    return sqlalchemy.create_engine(url, future=True)


def init_sql_schema(engine) -> None:
    """Apply ``schema.sql`` idempotently against a SQLAlchemy engine or DB URL string."""
    from importlib import resources

    try:
        import sqlalchemy
    except ModuleNotFoundError as exc:
        raise ImportError(
            "SQLAlchemy is required for Tier-2 database workflows. Install with: pip install -e '.[tier2]'"
        ) from exc

    owns_engine = isinstance(engine, str)
    eng = get_sqlalchemy_engine(engine) if owns_engine else engine
    ddl_sql = resources.files("cap_alerting_coverage.ingestion").joinpath("schema.sql").read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in ddl_sql.split(";") if stmt.strip()]
    try:
        with eng.begin() as conn:
            for stmt in statements:
                conn.execute(sqlalchemy.text(stmt))
    finally:
        if owns_engine:
            eng.dispose()


def get_optional_spanner_database(
    project_id: str | None = None,
    instance_id: str | None = None,
    database_id: str | None = None,
):
    """Create an optional Google Cloud Spanner database handle strictly from explicit env vars."""
    proj = project_id or os.environ.get("CAP_GAP_SPANNER_PROJECT_ID", "").strip()
    inst = instance_id or os.environ.get("CAP_GAP_SPANNER_INSTANCE_ID", "").strip()
    db_id = database_id or os.environ.get("CAP_GAP_SPANNER_DATABASE_ID", "").strip()
    if not (proj and inst and db_id):
        raise RuntimeError(
            "Cloud Spanner requires explicit CAP_GAP_SPANNER_PROJECT_ID, "
            "CAP_GAP_SPANNER_INSTANCE_ID, and CAP_GAP_SPANNER_DATABASE_ID."
        )
    from google.cloud import spanner

    client = spanner.Client(project=proj)
    return client.instance(inst).database(db_id)

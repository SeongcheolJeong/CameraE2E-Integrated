#!/usr/bin/env python3
"""Export per-company Lens Patent simulation DB sets for CameraE2E."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMERAE2E_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_DB = CAMERAE2E_ROOT / "src" / "pyisetcam" / "data" / "lens_patents" / "lens_patent_simulation_v6.sqlite"
DEFAULT_OUT_DIR = CAMERAE2E_ROOT / "src" / "pyisetcam" / "data" / "lens_patents" / "companies"
TABLES = ("metadata", "companies", "lenses", "lens_surfaces", "simulation_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-db", type=Path, default=DEFAULT_MASTER_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    master = sqlite3.connect(args.master_db)
    master.row_factory = sqlite3.Row
    schema = load_schema(master)
    companies = master.execute(
        """
        SELECT company, company_slug
        FROM companies
        ORDER BY company
        """
    ).fetchall()

    manifest_rows: list[dict[str, Any]] = []
    for company in companies:
        row = export_company(master, schema, args.out_dir, company, args.overwrite)
        manifest_rows.append(row)

    manifest = {
        "schema": "camerae2e_lens_patent_company_sets_v1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "master_db": str(args.master_db),
        "company_count": len(manifest_rows),
        "summary": summarize_manifest(manifest_rows),
        "companies": manifest_rows,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(args.out_dir / "README.md", manifest)
    master.close()
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


def export_company(
    master: sqlite3.Connection,
    schema: dict[str, list[str]],
    out_dir: Path,
    company: sqlite3.Row,
    overwrite: bool,
) -> dict[str, Any]:
    company_name = str(company["company"])
    company_slug = str(company["company_slug"])
    company_dir = out_dir / company_slug
    company_dir.mkdir(parents=True, exist_ok=True)
    db_path = company_dir / f"lens_patent_simulation_v6_{company_slug}.sqlite"
    if db_path.exists() and not overwrite:
        summary = summarize_company_db(db_path)
        return {
            "company": company_name,
            "company_slug": company_slug,
            "db": str(db_path.relative_to(out_dir)),
            "status": "exists",
            **summary,
        }

    tmp_path = db_path.with_suffix(".tmp.sqlite")
    if tmp_path.exists():
        tmp_path.unlink()
    target = sqlite3.connect(tmp_path)
    target.row_factory = sqlite3.Row
    create_schema(target, schema)

    copy_table(master, target, "companies", "company_slug = ?", (company_slug,))
    copy_table(master, target, "lenses", "company_slug = ?", (company_slug,))
    lens_ids = [
        str(row["lens_id"])
        for row in master.execute("SELECT lens_id FROM lenses WHERE company_slug = ?", (company_slug,))
    ]
    copy_lens_surfaces(master, target, lens_ids)
    copy_table(master, target, "simulation_results", "company_slug = ?", (company_slug,))
    write_metadata(master, target, company_name, company_slug)
    create_indexes(target, schema)
    summary = summarize_company_con(target)
    target.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("summary", json.dumps(summary, sort_keys=True)),
    )
    target.commit()
    target.close()
    tmp_path.replace(db_path)

    (company_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "company": company_name,
        "company_slug": company_slug,
        "db": str(db_path.relative_to(out_dir)),
        "status": "generated",
        **summary,
    }


def load_schema(con: sqlite3.Connection) -> dict[str, list[str]]:
    schema: dict[str, list[str]] = {"tables": [], "indexes": []}
    for row in con.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE sql IS NOT NULL
          AND name NOT LIKE 'sqlite_%'
        ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END, name
        """
    ):
        if row["type"] == "table" and row["name"] in TABLES:
            schema["tables"].append(str(row["sql"]))
        elif row["type"] == "index":
            schema["indexes"].append(str(row["sql"]))
    return schema


def create_schema(con: sqlite3.Connection, schema: dict[str, list[str]]) -> None:
    for sql in schema["tables"]:
        con.execute(sql)
    con.commit()


def create_indexes(con: sqlite3.Connection, schema: dict[str, list[str]]) -> None:
    for sql in schema["indexes"]:
        con.execute(sql)
    con.commit()


def copy_table(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    table: str,
    where: str,
    params: tuple[Any, ...],
) -> None:
    columns = table_columns(source, table)
    rows = source.execute(f"SELECT {', '.join(columns)} FROM {table} WHERE {where}", params).fetchall()
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    target.executemany(
        f"INSERT INTO {table}({', '.join(columns)}) VALUES ({placeholders})",
        ([row[column] for column in columns] for row in rows),
    )
    target.commit()


def copy_lens_surfaces(source: sqlite3.Connection, target: sqlite3.Connection, lens_ids: list[str]) -> None:
    if not lens_ids:
        return
    columns = table_columns(source, "lens_surfaces")
    placeholders = ", ".join("?" for _ in lens_ids)
    rows = source.execute(
        f"SELECT {', '.join(columns)} FROM lens_surfaces WHERE lens_id IN ({placeholders})",
        lens_ids,
    ).fetchall()
    if not rows:
        return
    insert_placeholders = ", ".join("?" for _ in columns)
    target.executemany(
        f"INSERT INTO lens_surfaces({', '.join(columns)}) VALUES ({insert_placeholders})",
        ([row[column] for column in columns] for row in rows),
    )
    target.commit()


def write_metadata(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    company: str,
    company_slug: str,
) -> None:
    for row in source.execute("SELECT key, value FROM metadata WHERE key = 'build_info'"):
        target.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (row["key"], row["value"]))
    target.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (
            "company_set",
            json.dumps(
                {
                    "company": company,
                    "company_slug": company_slug,
                    "built_at": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            ),
        ),
    )
    target.commit()


def table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [str(row["name"]) for row in con.execute(f"PRAGMA table_info({table})")]


def summarize_company_db(db_path: Path) -> dict[str, Any]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return summarize_company_con(con)
    finally:
        con.close()


def summarize_company_con(con: sqlite3.Connection) -> dict[str, Any]:
    def one(query: str) -> int:
        return int(con.execute(query).fetchone()[0])

    return {
        "lenses": one("SELECT count(*) FROM lenses"),
        "surfaces": one("SELECT count(*) FROM lens_surfaces"),
        "simulation_results": one("SELECT count(*) FROM simulation_results"),
        "camerae2e_ready": one(
            "SELECT count(*) FROM simulation_results WHERE simulation_status = 'camerae2e_ready'"
        ),
        "partial": one("SELECT count(*) FROM simulation_results WHERE simulation_status = 'partial'"),
        "metadata_only": one("SELECT count(*) FROM simulation_results WHERE simulation_status = 'metadata_only'"),
        "readiness_counts": {
            row["readiness"]: row["count"]
            for row in con.execute("SELECT readiness, count(*) AS count FROM lenses GROUP BY readiness")
        },
    }


def summarize_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "companies": len(rows),
        "lenses": sum(int(row.get("lenses", 0)) for row in rows),
        "surfaces": sum(int(row.get("surfaces", 0)) for row in rows),
        "simulation_results": sum(int(row.get("simulation_results", 0)) for row in rows),
        "camerae2e_ready": sum(int(row.get("camerae2e_ready", 0)) for row in rows),
        "partial": sum(int(row.get("partial", 0)) for row in rows),
        "metadata_only": sum(int(row.get("metadata_only", 0)) for row in rows),
    }


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Lens Patent Company DB Sets",
                "",
                "Each subdirectory contains a company-specific SQLite subset generated from "
                "`lens_patent_simulation_v6.sqlite`.",
                "",
                "Regenerate:",
                "",
                "```bash",
                "python tools/export_lens_patent_company_sets.py --overwrite",
                "```",
                "",
                "Use a company DB directly:",
                "",
                "```python",
                "from pyisetcam.lens_patents import lens_patent_search, lens_patent_optics",
                "",
                "db_path = 'src/pyisetcam/data/lens_patents/companies/canon/lens_patent_simulation_v6_canon.sqlite'",
                "row = lens_patent_search(db_path=db_path, require_camerae2e=True, limit=1)[0]",
                "optics = lens_patent_optics(row['simulation_id'], db_path=db_path)",
                "```",
                "",
                f"Summary: `{json.dumps(manifest['summary'], sort_keys=True)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())

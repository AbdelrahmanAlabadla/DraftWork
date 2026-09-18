from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    migrations = sorted(Path("migrations").glob("*.sql"))
    if not migrations:
        raise RuntimeError("No SQL migrations were found")
    with psycopg.connect(database_url) as connection:
        for migration in migrations:
            connection.execute(migration.read_text(encoding="utf-8"), prepare=False)
        connection.commit()
    print(f"Applied {len(migrations)} migration file(s)")


if __name__ == "__main__":
    main()

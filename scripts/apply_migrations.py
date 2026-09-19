from __future__ import annotations

import os

from alembic import command
from alembic.config import Config
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    alembic_config = Config("alembic.ini")
    command.upgrade(alembic_config, "head")
    print("Database is at the latest Alembic revision")


if __name__ == "__main__":
    main()

from pathlib import Path


def test_api_worker_and_migration_use_one_explicit_container_database_url():
    compose = (Path(__file__).parents[1] / "compose.yaml").read_text(encoding="utf-8")
    canonical = (
        "DATABASE_URL: ${DOCKER_DATABASE_URL:?Set DOCKER_DATABASE_URL in .env "
        "to the canonical PostgreSQL instance}"
    )
    assert compose.count(canonical) == 3
    assert "DATABASE_URL: postgresql://" not in compose


def test_example_distinguishes_host_and_container_addresses_without_changing_database():
    env = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")
    assert "DATABASE_URL=postgresql://postgres:your_password@localhost:1966/draftwork" in env
    assert (
        "DOCKER_DATABASE_URL=postgresql://postgres:your_password@"
        "host.docker.internal:1966/draftwork"
    ) in env

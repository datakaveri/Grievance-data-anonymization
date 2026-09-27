# Changelog

All notable changes to `Grievance-data-anonymization` will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-25

### Added
- Standardized package architecture matching required repository rules (`src/grievance_anonymization/`).
- GitHub Flow CI/CD workflows (`ci.yml`, `release.yml`, `codeql.yml`).
- Unit and integration test suite (`tests/unit/`, `tests/integration/`).
- Open source Apache-2.0 `LICENSE` and security policy `SECURITY.md`.
- `pyproject.toml` and pinned `requirements.lock` for reproducible builds.
- Docker non-root user execution (`appuser`) and container `HEALTHCHECK`.
- Pre-commit configuration with Ruff, Black, and GitLeaks secret scanning.

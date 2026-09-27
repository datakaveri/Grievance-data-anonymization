# Contributing Guidelines

Thank you for contributing to the Anonymisation Repositories suite!

## Branching & Commit Policy

We enforce **GitHub Flow**:
- Default branch is `main` (protected, requiring passing CI).
- Create short-lived feature or fix branches following kebab-case naming:
  - `feat/<short-kebab-description>`
  - `fix/<short-kebab-description>`
  - `refactor/<short-kebab-description>`
  - `docs/<short-kebab-description>`
  - `chore/<short-kebab-description>`

### Prohibited Branch Names
- No uppercase letters or underscores.
- No version numbers in branch names (use Git tags `vX.Y.Z` for releases).
- No status words like `final`, `latest`, `updated`, or `testing`.

## Commit Messages

Follow Conventional Commits:
`feat: add support for custom PII regex rules`
`fix: correct character offset calculation on multiline text`

## Development Workflow

1. Install dependencies and package in editable mode:
   ```bash
   pip install -e .[dev]
   ```
2. Run tests and linting:
   ```bash
   pytest
   ruff check .
   black --check .
   ```
3. Submit a Pull Request targeting `main`.

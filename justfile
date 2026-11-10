# -- Settings ------------------------------------------------------------------

# Use latest version of PowerShell on Windows
set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]

# -- Recipes -------------------------------------------------------------------

# Setup Python environment
[group('setup')]
setup:
	uv venv --allow-existing
	uv sync --all-extras

# Check code with various linters
[group('lint')]
check: setup
	uv run ruff check

# Run tests (without the slow ones)
[default]
[group('test')]
test: check
	uv run pytest -m "not slow"

# Run all tests, including large files and long fuzz runs
[group('test')]
test-all: check
	uv run pytest

# Rewrite the stored synthetic test files (tests/data/synthetic) from tests/testcases.py; a folder may be given instead
[group('test')]
testcases folder="tests/data/synthetic": setup
	uv run python -m tests.testcases_cli {{folder}}

# Regenerate doc/special-cases.md from the catalogue in tests/catalogue.py
[group('test')]
catalogue: setup
	uv run python -m tests.gen_catalogue_doc

# Regenerate the expected values of the real files in tests/data/real (only with a reader that has been verified)
[group('test')]
real-expected: setup
	uv run python -m tests.gen_real_expected

# Release new package version
[group('release')]
[unix]
release version:
	#!/usr/bin/env sh -e
	uv version {{version}}
	version="$(uv version --short)"
	git commit -a -m "Release: Release version ${version}"
	git tag "${version}"
	git push
	git push --tags

# Release new package version
[group('release')]
[windows]
release version:
	#!pwsh
	uv version {{version}}
	set version "$(uv version --short)"
	git commit -a -m "Release: Release version ${version}"
	git tag "${version}"
	git push
	git push --tags

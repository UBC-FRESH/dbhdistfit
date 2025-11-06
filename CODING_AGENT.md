# Codex Agent Operating Notes

When contributing to this repository as the coding agent:

1. Before wrapping up a development milestone (feature, roadmap phase, or PR), run:
   - `ruff format src tests`
   - `ruff check src tests`
   - `mypy src`
   - `pytest`
   - `pre-commit run --all-files` (once hooks are installed).
2. Address any linter or type-check warnings promptly; avoid suppressions unless discussed.
3. Prefer module-level constants for Typer argument defaults (prevents B008) and keep line length
   ≤ 100 characters.
4. Document notable behaviour changes in the README or docs as part of the same change set.
5. Whenever you plan or complete work, update the “Detailed Next Steps Notes” section in
   `ROADMAP.md` so the leading edge of the implementation plan stays current. Consult that section
   before proposing new next steps to ensure we continue in sequence rather than jumping around.

Treat these steps as the minimum bar for every milestone so manual reminders are not required.

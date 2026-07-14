/**
 * Commitlint config — DraftForge.
 *
 * Enforces Conventional Commits + 72-char subject line at commit-time.
 * This is the LOCAL pre-commit contract; the CI workflow (.github/workflows/ci.yml)
 * uses a regex pattern that is a subset of this. The local config is the
 * source of truth for human-readable rules.
 *
 * Why a JS config: standard `@commitlint/config-conventional` extends
 * rules; we override header-max-length + body-max-line-length to align
 * with CLAUDE.md §"Git + PR Hygiene" ("Subject ≤72 chars, imperative, no period").
 *
 * Install: `npm install --save-dev @commitlint/cli @commitlint/config-conventional husky`
 * Then in package.json:
 *   "husky": { "hooks": { "commit-msg": "commitlint -E HUSKY_GIT_PARAMS" } }
 *
 * Until husky is wired, `make commitlint` (added below) validates HEAD commit.
 */
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'header-max-length': [2, 'always', 72],
    'body-max-line-length': [2, 'always', 100],
    'subject-case': [2, 'always', 'lower-case'],
    'subject-empty': [2, 'never'],
    'subject-full-stop': [2, 'never', '.'],
    'type-case': [2, 'always', 'lower-case'],
    'type-empty': [2, 'never'],
  },
};
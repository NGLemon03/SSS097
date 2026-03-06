# Git Backup Workflow

## Goal
Keep this repository focused on source code and docs, while backing up to both local Git history and a private GitHub remote.

## Default Rules
- Backup-pattern files are ignored by default: `*.bak`, `*.corrupt_backup`, `*.pre_rebuild_*`, `backup_*`, `*_backup_*`.
- Data/cache/output files are ignored by default (`data/`, `results/`, `cache/`, `*.db`, etc.).
- If a backup file must be preserved in Git, add it explicitly with force:
  - `git add -f <path-to-backup-file>`

## First-Commit Time Cutoff Rule
- Keep only backup-pattern files modified in the last 14 days in the initial commit.
- Backup-pattern files older than 14 days stay in working directory and are not committed.

## Daily Backup Flow
1. Check status:
   - `git status --short --ignored`
2. Stage normal code/docs:
   - `git add .`
3. Optionally include specific backup files:
   - `git add -f <file>`
4. Commit:
   - `git commit -m "backup: <short note>"`
5. Push to GitHub:
   - `git push`

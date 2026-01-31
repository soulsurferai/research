# Research

This repository is the public **methods + artifacts** companion to the Noetheca portfolio.

It contains working files behind published reports (data extracts, notebooks, scripts, generated outputs). Some folders are messy by design: they reflect real iteration and learning.

## How it’s organized

### Report folders (one per portfolio report)

Each folder at the repo root (e.g. `/folk-horror`, `/nyfa-youtube`) is a self-contained bundle of **the most relevant work artifacts** for that report.

### Shared pipelines (reusable code)

Reusable/agnostic code lives under:

- `/_pipelines/`

This avoids duplicating the same notebooks/scripts across multiple report folders.

## Report index

- Folk Horror: `folk-horror/`
- NYFA YouTube: `nyfa-youtube/`
- Cultural Analytics (WorldviewV2): `cultural-analytics/`
- LOTR: `lotr/`
- Cannabis Communities: `cannabis-communities/`

## Notes

- Credentials are not stored in this repo. Scripts load secrets via environment variables and/or external `.env` files.
- Large archives and caches are generally excluded.
- Source material (e.g. Reddit comments) may include strong language; it is included for research provenance.

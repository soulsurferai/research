# NYFA YouTube (work files)

This folder contains the working artifacts for the NYFA + Film Courage YouTube competitive intelligence analysis.

## Contents

- `data/` — channel exports (CSV)
- `notebooks/` — analysis notebooks + generated outputs (charts/JSON/CSVs)
- `transcripts/` — transcript runs/output (when used)
- `youtube_data_extraction.py` — YouTube Data API extraction script
- `transcript.py` — transcript helper script

## Running the extractor

This repo intentionally does **not** include API keys.

Set an environment variable and run:

```bash
export YT_API_KEY="..."
python youtube_data_extraction.py --channelname "NYFA"
```


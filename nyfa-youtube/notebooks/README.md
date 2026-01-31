# YouTube Analysis Notebooks

## Directory Structure

```
notebooks/
├── README.md                          # This file
├── youtube_analysis_v1.ipynb          # Current working notebook
├── output/                            # LLM-optimized analysis outputs
│   ├── analysis_YYYYMMDD.md          # Markdown reports
│   ├── insights_YYYYMMDD.json        # Structured data
│   └── charts/                       # Exported visualizations
├── archive/                          # Previous notebook versions
│   └── youtube_analysis_v1_YYYYMMDD.ipynb
└── templates/                        # Reusable notebook templates
```

## Workflow

### 1. Running Analysis
- Use `youtube_analysis_v1.ipynb` for all current work
- Notebook automatically loads data from `../data/`
- Execute cells sequentially or use "Run All"

### 2. Iteration Process
When making significant changes:
1. Copy current notebook to `archive/` with datestamp
2. Make changes to main notebook
3. Document changes in notebook's changelog cell

### 3. LLM-Optimized Output

The notebook generates output specifically designed for LLM analysis:

**Markdown Reports (`output/analysis_YYYYMMDD.md`):**
- Executive summary with key metrics
- Section headers for easy parsing
- Data tables in markdown format
- Key insights highlighted
- Actionable recommendations

**JSON Data (`output/insights_YYYYMMDD.json`):**
- Structured metrics for programmatic analysis
- Time-series data
- Comparative statistics
- Top performers lists

**Chart Exports (`output/charts/`):**
- PNG files of key visualizations
- Filenames describe content: `publishing_velocity_comparison.png`
- Can be referenced in LLM conversations

### 4. Output Format Guidelines

**For optimal LLM analysis, outputs follow these principles:**

1. **Hierarchical Structure**: Clear headers (##, ###) for easy navigation
2. **Context-Rich**: Each section includes context, not just raw numbers
3. **Actionable**: Every insight paired with "so what?" interpretation
4. **Comparable**: Consistent formatting across time periods/channels
5. **Self-Contained**: Reports include enough context to be understood standalone

**Example Output Structure:**
```markdown
# YouTube Competitive Analysis Report
Generated: 2025-12-13

## Executive Summary
[3-5 key findings with context]

## Channel Comparison Matrix
| Metric | NYFA | Film Courage | Insight |
|--------|------|--------------|---------|
[Comparative data with interpretation]

## Deep Dive: Publishing Strategy
### Current State
[Description with numbers]

### Trend Analysis
[What's changing and why it matters]

### Recommendations
[Specific, actionable steps]
```

## Notebook Design Principles

### For Easy Iteration
- **Modular cells**: Each analysis is self-contained
- **Reusable functions**: Common operations defined once
- **Clear comments**: Non-obvious logic explained
- **Parameter cells**: Easy to change date ranges, channels, thresholds
- **Validation outputs**: Print statements confirm data loaded correctly

### For LLM Analysis
- **Progressive disclosure**: Build from overview to detail
- **Explicit calculations**: Show the math, don't hide it
- **Contextual tables**: Include column descriptions
- **Insight annotations**: Markdown cells interpret results
- **Linked references**: Connect findings to source data

## File Naming Conventions

**Notebooks:**
- Working version: `youtube_analysis_v1.ipynb`
- Archived versions: `youtube_analysis_v1_20251213.ipynb`
- Specialized analyses: `youtube_cta_analysis_v1.ipynb`

**Output Files:**
- Reports: `analysis_20251213.md`
- Data: `insights_20251213.json`
- Charts: `{metric}_{comparison}_{date}.png`
  - Example: `views_per_day_comparison_20251213.png`

## Adding New Channels

When new channel data is extracted:
1. CSV automatically saved to `../data/`
2. Notebook detects new files automatically (via glob pattern)
3. Re-run analysis cells to include new data
4. Generate updated output with new datestamp

## Version Control Strategy

- Main notebook: Iterative improvements, keep history in archive
- Output files: Datestamped, never overwrite
- Data files: Append new extractions, keep all historical data
- Archive: Only when making breaking changes to analysis logic

## Common Analyses

The notebook supports these standard analyses out-of-box:
- [ ] Channel overview comparison
- [ ] Publishing velocity trends
- [ ] Time-normalized performance (views/day)
- [ ] Engagement rate analysis
- [ ] Content length optimization
- [ ] CTA strategy analysis
- [ ] Top performers identification
- [ ] Temporal trend analysis
- [ ] Strategic recommendations generation

## Next Steps

1. Create initial `youtube_analysis_v1.ipynb`
2. Test with NYFA data
3. Add Film Courage data and re-run
4. Generate first markdown report
5. Feed to LLM for strategic insights
6. Iterate based on findings

# LOTR Reddit Signal Extraction

Computational analysis of Lord of the Rings fan discourse on Reddit, extracting actionable signals for game development strategy. Built for Embracer Group / Eidos consideration.

## What This Is

A semantic research investigation mining **69,037 Reddit comments** and **5,308 posts** from LOTR-related subreddits to identify what fans actually want, love, fear, and argue about — extracted as structured, analyzable signals rather than anecdotal cherry-picks.

## Methodology

**Rapid Semantic Discovery** — combining SQL sampling, AI-assisted pattern recognition, and the "Rule of Three" validation:

1. **Statistical signal** — does it show up in the numbers?
2. **Textual validation** — do real comments confirm the pattern?
3. **Behavioral confirmation** — do users act on it (upvotes, engagement, cross-posting)?

### Pipeline

1. Source data pulled from Supabase (PostgreSQL) via Python/psycopg2
2. Entity extraction using custom knowledge graphs (spaCy EntityRuler patterns)
3. Intent classification: OPINION, PAIN_POINT, NOSTALGIA, WISHLIST, HYPE
4. Sentiment scoring with domain-aware tuning
5. Cross-subreddit analysis for external signal validation

## Key Findings

Extracted **1,881 structured signals** across categories:

- **Wishlist signals** (234) — what fans explicitly want in a LOTR game
- **Praise signals** (1,155) — what fans love about existing adaptations
- **Temporal analysis** (27) — how sentiment shifts over time
- **External discourse** (4,269 comments) — LOTR discussion outside fan subreddits

See `slides/` for visual summaries.

## Repository Structure

```
├── notebooks/
│   ├── LOTR_Reddit_Signal_Extraction_v1.ipynb    # Initial investigation
│   └── LOTR_Reddit_Signal_Extraction_v2.ipynb    # Refined analysis
├── scripts/
│   ├── igdb.py              # IGDB API integration
│   ├── vg_curate.py         # Video game entity curation
│   └── vg_gap.py            # Gap analysis tooling
├── entities/
│   ├── lotr_entities.json   # LOTR character/location/concept entities
│   ├── vg_entities.json     # Video game entity definitions
│   ├── vg_aesthetics.json   # Visual/artistic style taxonomy
│   ├── vg_mechanics.json    # Gameplay mechanics taxonomy
│   └── vg_tech.json         # Technical feature taxonomy
├── output/
│   ├── lotr_comments_69k.csv                    # Source: 69,037 Reddit comments
│   ├── lotr_posts_5k.csv                        # Source: 5,308 Reddit posts
│   ├── Embracer_Reddit_Signals_MASTER_v1.csv    # 1,881 extracted signals
│   ├── wishlist_signals.csv                     # Wishlist intent subset
│   ├── praise_signals.csv                       # Praise intent subset
│   ├── external_sub_comments.csv                # Cross-subreddit mentions
│   ├── tales_full.csv                           # Tales of Middle-earth analysis
│   └── temporal_analysis.csv                    # Sentiment over time
├── slides/
│   ├── Slide1_Real_Wishlist.png
│   ├── Slide2_Pain_Points.png
│   └── Slide3_Implicit_Love.png
└── requirements.txt
```

## Data Source

Comments and posts collected from Reddit via automated pipeline into Supabase PostgreSQL (6.9M total comments in database; 69K LOTR-specific subset extracted for this analysis). Collection ongoing via Edge Functions with cron scheduling.

## Tech Stack

- **Database:** PostgreSQL (Supabase) with pgvector for semantic search
- **Analysis:** Python, pandas, psycopg2
- **NLP:** spaCy, custom EntityRuler patterns, NLTK
- **Embeddings:** OpenAI text-embedding-3-large (1536 dimensions)
- **Notebooks:** Jupyter (localhost)

## Author

**James Root** — [Noetheca](https://github.com/soulsurferai/noetheca)

Digital marketing and data analytics professional. 30+ years experience at Microsoft, Electronic Arts, and 20th Century Studios. Now focused on applying enterprise-scale analytical tools to cultural and civic research.

## License

Private research. All Reddit data subject to Reddit API terms of service.

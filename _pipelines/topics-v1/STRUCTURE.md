# Project Structure - Topics_V1

## 📁 Directory Structure

```
Topics_V1/
├── README.md                 # Project overview and results
├── requirements.txt          # Python dependencies
├── config.py                # Centralized configuration
├── run_analysis.py          # Main entry point
├── tasklist.md              # Development task tracking
│
├── core/                    # Core analysis scripts
│   ├── topics_advanced.py   # Main topic analysis pipeline
│   ├── create_gartner_matrix.py      # Community positioning visualization
│   ├── create_topic_comparison.py    # Topic distribution charts
│   └── generate_insights_report.py   # Strategic insights report
│
├── utils/                   # Reusable utilities
│   ├── __init__.py
│   ├── json_utils.py       # JSON encoding with numpy support
│   ├── qdrant_utils.py     # Qdrant vector database interface
│   └── text_utils.py       # Reddit text preprocessing
│
├── analysis/               # Analysis algorithms
│   ├── __init__.py
│   ├── clustering.py       # Basic clustering algorithms
│   ├── enhanced_clustering.py        # Advanced UMAP + HDBSCAN
│   ├── enhanced_topic_extraction.py  # BM25, MMR, coherence scoring
│   ├── topic_extraction.py           # Basic topic extraction
│   ├── sentiment_analysis.py         # Sentiment analysis tools
│   ├── advanced_nlp_tools.py         # Advanced NLP utilities
│   ├── algorithm_improvements.py     # Algorithm enhancements
│   │
│   ├── nlp/               # Specialized NLP modules
│   │   ├── __init__.py
│   │   └── metaphor.py    # Metaphor detection and analysis
│   │
│   ├── extraction/        # Topic extraction methods (TO BE CREATED)
│   └── metrics/          # Evaluation metrics (TO BE CREATED)
│
├── scripts/              # Utility and experimental scripts
│   ├── debug/           # Debug scripts (marijuana case issues)
│   │   ├── debug_marijuana.py
│   │   ├── check_marijuana_case.py
│   │   ├── check_marijuana_data.py
│   │   ├── marijuana_investigation.py
│   │   ├── quick_marijuana_check.py
│   │   ├── simple_marijuana_check.py
│   │   └── test_marijuana_direct.py
│   │
│   └── experiments/     # Experimental analysis scripts
│       ├── analyze_label_issues.py
│       ├── analyze_other_topics.py
│       ├── analyze_reddit_metaphors.py
│       ├── analyze_sample_sizes.py
│       ├── check_lengths.py
│       ├── fix_topic_labels.py
│       └── run_full_metaphor_analysis.py
│
├── tests/               # Test suite
│   ├── test_both_cases.py
│   └── test_metaphor_detection.py
│
├── archive/             # Archived/old versions
│   ├── topics.py        # Original implementation
│   ├── topics_simple.py # Simplified version
│   ├── topics_modular.py # Modular refactor
│   ├── refactoring_plan.py
│   └── run_fix_labels.sh
│
└── results/             # Analysis outputs (renamed from quick_results)
    ├── analysis_summary_advanced.json
    ├── subreddit_comparison.csv
    ├── cannabis_community_insights.md
    ├── cannabis_gartner_matrix.png
    ├── topic_theme_comparison.png
    └── [subreddit]_topics_advanced.csv (for each community)
```

## 🚀 Quick Start

```bash
# Run full analysis on all subreddits
python run_analysis.py

# Analyze specific subreddit
python run_analysis.py --subreddit cannabis

# Run with visualizations and report
python run_analysis.py --visualize --report

# Generate visualizations from existing data
python run_analysis.py --skip-analysis --visualize --report

# Custom sample size
python run_analysis.py --samples 1000
```

## 📊 Available Analysis Options

- **Topic Analysis**: Advanced BERTopic with UMAP + HDBSCAN
- **Visualizations**: Gartner matrix and topic comparisons
- **Reports**: Strategic insights and recommendations
- **Metaphor Analysis**: Conceptual metaphor detection
- **Sentiment Analysis**: Community sentiment patterns

## 🔧 Next Steps for Enhanced Insights

1. **Refactor `enhanced_topic_extraction.py`** (400+ lines)
   - Split into `extraction/` modules
   - Separate BM25, MMR, KeyBERT methods
   - Create dedicated metrics modules

2. **Add Temporal Analysis**
   - Track topic evolution over time
   - Identify emerging vs declining themes
   - Seasonal patterns in discussions

3. **Cross-Community Analysis**
   - Identify semantic bridges between communities
   - Find universal vs community-specific topics
   - Map ideological distances

4. **Enhanced Metaphor Analysis**
   - Deeper conceptual metaphor patterns
   - Community-specific metaphor usage
   - Metaphor evolution tracking

5. **Network Analysis**
   - User interaction patterns
   - Information flow between communities
   - Influence and authority mapping

## 📝 Configuration

All settings in `config.py`:
- Qdrant connection details
- Analysis parameters
- Text processing options
- Output directories

## 🧪 Testing

Run tests with:
```bash
python -m pytest tests/
```

## 📈 Performance Notes

- Tested with 4,244 documents across 6 subreddits
- Memory efficient with batch processing
- Scalable to larger datasets with parameter tuning

# Noetheca Subreddit Worldview Analyzer

This collection of scripts provides a powerful tool for analyzing and comparing the semantic "worldviews" of different subreddit communities. It leverages the Qdrant vector database containing embedded Reddit comments from the Noetheca project to discover how different communities frame and discuss similar topics.

## Overview

The system consists of four main components:

1. **Reddit Chooser** (`reddit_chooser.py`): Lists available subreddits and lets you select which ones to analyze
2. **Worldview Analyzer** (`worldview_analyzer.py`): Performs in-depth semantic analysis on the selected subreddits
3. **Emotion Embeddings** (`emotion_embeddings.py`): Provides advanced emotion analysis using OpenAI embeddings
4. **Run Analysis** (`run_analysis.py`): A wrapper script that coordinates the workflow and generates AI-ready prompts

## Features

- **Subreddit Discovery**: Automatically lists all available subreddits in your Qdrant database
- **Topic Filtering**: Focus analysis on specific topics (e.g., "economy", "healthcare", "immigration")
- **Multi-dimensional Analysis**:
  - Semantic distance between communities (Jensen-Shannon divergence)
  - Distinctive concepts for each community
  - Emotional profiles (lexical and embedding-based)
  - Potential bridge concepts
  - Visualization of semantic space
- **Hybrid Emotion Analysis**:
  - Lexical pattern matching for basic emotion detection
  - Advanced embedding-based emotion analysis using OpenAI API
  - Identification of emotional bridges between communities
- **AI-Ready Output**: Generates prompts for AI interpretation of the analysis results

## Setup

### Prerequisites

- Python 3.6+
- Access to a Qdrant database containing Reddit comments with semantic embeddings
- OpenAI API key (for the embedding-based emotion analysis)
- Required Python packages:
  ```
  pip install qdrant-client numpy pandas matplotlib seaborn scikit-learn python-dotenv requests
  ```

### Environment Configuration

The scripts expect to find your `.env` file in an `env` directory that's a sibling to your `Worldview` folder:

```
/your_base_directory
├── env
│   └── .env
├── Worldview
│   ├── reddit_chooser.py
│   ├── worldview_analyzer.py
│   ├── emotion_embeddings.py
│   └── run_analysis.py
└── embeddings_cache  # Will be created automatically
```

Your `.env` file should contain:

```
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Quick Start

The easiest way to run the complete workflow is to use the wrapper script:

```bash
python run_analysis.py
```

This will:
1. List available subreddits
2. Let you select two subreddits to compare
3. Run the analysis (including both lexical and embedding-based emotion analysis)
4. Save results and visualizations
5. Generate an AI-ready prompt for interpretation

### Step-by-Step Workflow

If you prefer to run each step separately:

1. **List and select subreddits**:
   ```bash
   python reddit_chooser.py
   ```

2. **Run analysis on selected subreddits**:
   ```bash
   python worldview_analyzer.py
   ```

3. **Generate an AI interpretation prompt from the latest results**:
   ```bash
   python run_analysis.py --prompt-only
   ```

### Advanced Options

- **Skip subreddit selection**:
  ```bash
  python run_analysis.py --skip-chooser
  ```

- **Skip analysis and only generate a prompt from existing results**:
  ```bash
  python run_analysis.py --skip-analyzer
  ```

## Output

The analysis produces several outputs:

1. **JSON Results**: Complete analysis data saved to `analysis_results/analysis_[subreddit_a]_vs_[subreddit_b]_[timestamp].json`
2. **Visualization**: 2D plot showing the semantic space of both subreddits, saved to `analysis_results/worldview_comparison_[run_id].png`
3. **AI Prompt**: Text file with findings formatted for AI interpretation, saved to `analysis_results/prompt_analysis_[run_id].json`
4. **Emotion Embeddings Cache**: OpenAI embeddings for emotion concepts are cached at `embeddings_cache/emotion_embeddings.json`

## Understanding the Results

### Semantic Distance

The Jensen-Shannon distance (0-1) measures how differently two communities "think" about topics. Higher values indicate more divergent worldviews.

### Distinctive Concepts

These are terms that are statistically overrepresented in one community compared to the other, revealing their unique focus or concerns.

### Emotional Profiles

Two types of emotion analysis are performed:

1. **Lexical Emotion Analysis**: Based on counting emotion-related keywords in the text
2. **Embedding-Based Emotion Analysis**: Uses OpenAI embeddings to perform more sophisticated semantic analysis of emotional content

The embedding-based analysis can detect implicit emotional content that isn't explicitly stated with emotion words, providing deeper insights into the emotional tone of communities.

### Worldview Bridges

Identifies concepts and ideas that have similar semantic meanings across both communities, potentially serving as common ground for communication.

## AI Interpretation

The generated prompt contains a structured summary of findings that can be fed to Claude or another AI for interpretation. This allows for deeper insight into:

1. Defining characteristics of each community's worldview
2. Emotional and cognitive differences
3. Potential bridges for cross-community communication
4. Underlying values explaining semantic differences
5. How these differences might affect information interpretation

## Embedding-Based Emotion Analysis

The system uses two complementary approaches to emotion analysis:

1. **Lexical Analysis**: Searches for explicit emotion words in the text (e.g., "angry", "happy")
2. **Embedding Analysis**: Uses OpenAI's text-embedding-3-large model to detect emotional content based on semantic meaning

The embedding-based approach:
- Can detect implicit emotions not explicitly stated with emotion words
- Uses rich descriptions of emotional states for comparison
- Identifies emotional bridges between communities
- Provides more nuanced analysis of emotional differences

Emotion embeddings are cached locally to avoid unnecessary API calls in future runs.

## Notes for Noetheca Project

This tool is specifically designed to work with the Qdrant database structure from the Noetheca project, with:

- Collection name: "reddit_comments"
- Vector field name: "semantic"
- Payload fields including "subreddit" and "content"

If your structure differs, you may need to adjust the code accordingly.

## Troubleshooting

For detailed troubleshooting information, see the `TROUBLESHOOTING.md` file.

## Contributing

Feel free to extend this system with additional analyses or visualizations. Some ideas:
- Temporal analysis to track how worldviews evolve over time
- Topic modeling to identify sub-themes within communities
- Network analysis of semantic relationships
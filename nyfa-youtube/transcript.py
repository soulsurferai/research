#!/usr/bin/env python3
"""
YouTube Transcript Extraction Script
Usage: 
  Single video: python transcript.py --video rYx4CjnpBGQ
  Batch from CSV: python transcript.py --input data/Film_Courage_12142024.csv
  Batch from text file: python transcript.py --input video_ids.txt
"""

import argparse
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from datetime import datetime
import os
import time
import re

def clean_transcript_text(transcript_data):
    """
    Convert transcript data to clean text optimized for LLM/NLP.
    Removes timestamps, filters artifacts, preserves paragraph structure.
    """
    # Join all text segments
    text = ' '.join([snippet.text for snippet in transcript_data.snippets])
    
    # Remove common YouTube artifacts
    artifacts = [
        r'\[Music\]',
        r'\[Applause\]',
        r'\[Laughter\]',
        r'\[Silence\]',
        r'\[Background noise\]',
        r'\[.*?\]'  # Any other bracketed content
    ]
    
    for pattern in artifacts:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = text.strip()
    
    # Add paragraph breaks at natural pauses (approximate)
    # YouTube auto-captions typically have segments, we'll group them
    paragraphs = []
    current_para = []
    
    for snippet in transcript_data.snippets:
        current_para.append(snippet.text)
        
        # Create paragraph break after ~30 seconds or at sentence boundaries
        if (snippet.duration > 5 and 
            any(snippet.text.rstrip().endswith(p) for p in ['.', '!', '?'])):
            para_text = ' '.join(current_para)
            # Clean artifacts from paragraph
            for pattern in artifacts:
                para_text = re.sub(pattern, '', para_text, flags=re.IGNORECASE)
            para_text = re.sub(r'\s+', ' ', para_text).strip()
            if para_text:  # Only add non-empty paragraphs
                paragraphs.append(para_text)
            current_para = []
    
    # Add any remaining text
    if current_para:
        para_text = ' '.join(current_para)
        for pattern in artifacts:
            para_text = re.sub(pattern, '', para_text, flags=re.IGNORECASE)
        para_text = re.sub(r'\s+', ' ', para_text).strip()
        if para_text:
            paragraphs.append(para_text)
    
    # Join paragraphs with double newline
    formatted_text = '\n\n'.join(paragraphs)
    
    return formatted_text

def get_transcript(video_id, language='en'):
    """
    Fetch transcript for a single video.
    Returns tuple: (success, transcript_text, metadata, error_message)
    """
    try:
        # Initialize API
        ytt_api = YouTubeTranscriptApi()
        
        # Get transcript list to check available languages
        transcript_list = ytt_api.list(video_id)
        
        # Try to get requested language, fall back to any available
        try:
            transcript_data = ytt_api.fetch(video_id, languages=[language])
            actual_language = language
            # Check if auto-generated
            auto_generated = transcript_data.is_generated
        except:
            # Get first available transcript
            if transcript_list:
                first_transcript = transcript_list[0]
                actual_language = first_transcript['language_code']
                transcript_data = ytt_api.fetch(video_id, languages=[actual_language])
                auto_generated = transcript_data.is_generated
            else:
                return False, None, None, "No transcripts available"
        
        # Clean and format transcript
        transcript_text = clean_transcript_text(transcript_data)
        
        # Calculate metadata
        word_count = len(transcript_text.split())
        
        metadata = {
            'transcript_retrieved': True,
            'transcript_length': word_count,
            'language': actual_language,
            'auto_generated': auto_generated,
            'retrieval_timestamp': datetime.now().isoformat(),
            'error_message': None
        }
        
        return True, transcript_text, metadata, None
        
    except TranscriptsDisabled:
        return False, None, None, "Transcripts disabled for this video"
    except NoTranscriptFound:
        return False, None, None, "No transcript found"
    except VideoUnavailable:
        return False, None, None, "Video unavailable"
    except Exception as e:
        return False, None, None, f"Error: {str(e)}"

def process_single_video(video_id, output_dir, language='en'):
    """Process a single video and save transcript."""
    print(f"\nProcessing video: {video_id}")
    
    success, transcript_text, metadata, error = get_transcript(video_id, language)
    
    if success:
        # Save transcript to file
        transcript_file = os.path.join(output_dir, f"{video_id}.txt")
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        
        print(f"✓ Success: {metadata['transcript_length']} words, {metadata['language']}, "
              f"{'auto-generated' if metadata['auto_generated'] else 'manual'}")
        print(f"  Saved to: {transcript_file}")
        
        return {
            'video_id': video_id,
            **metadata
        }
    else:
        print(f"✗ Failed: {error}")
        return {
            'video_id': video_id,
            'transcript_retrieved': False,
            'transcript_length': 0,
            'language': None,
            'auto_generated': None,
            'retrieval_timestamp': datetime.now().isoformat(),
            'error_message': error
        }

def process_batch(video_ids, output_dir, language='en', delay=1):
    """
    Process multiple videos.
    Returns DataFrame with results.
    """
    results = []
    total = len(video_ids)
    
    for idx, video_id in enumerate(video_ids, 1):
        print(f"\n[{idx}/{total}] Processing: {video_id}")
        
        result = process_single_video(video_id, output_dir, language)
        results.append(result)
        
        # Rate limiting
        if idx < total and delay > 0:
            time.sleep(delay)
    
    return pd.DataFrame(results)

def read_input_file(input_path):
    """
    Read video IDs from CSV or text file.
    Returns list of video IDs.
    """
    if input_path.endswith('.csv'):
        # Read CSV and extract video_id column
        df = pd.read_csv(input_path)
        
        if 'video_id' not in df.columns:
            raise ValueError("CSV must contain 'video_id' column")
        
        video_ids = df['video_id'].dropna().unique().tolist()
        print(f"Loaded {len(video_ids)} unique video IDs from CSV")
        
    else:
        # Assume text file with one video ID per line
        with open(input_path, 'r') as f:
            video_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(video_ids)} video IDs from text file")
    
    return video_ids

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract YouTube video transcripts')
    parser.add_argument('--video', help='Single video ID to process')
    parser.add_argument('--input', help='CSV or text file with video IDs')
    parser.add_argument('--output-dir', default='transcripts', 
                       help='Output directory (default: transcripts)')
    parser.add_argument('--language', default='en', 
                       help='Preferred language code (default: en)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between requests in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video and not args.input:
        parser.error("Must provide either --video or --input")
    
    if args.video and args.input:
        parser.error("Cannot use both --video and --input")
    
    print(f"\n{'='*60}")
    print(f"YouTube Transcript Extraction")
    print(f"{'='*60}\n")
    
    # Create output directory structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output_dir):
        output_base = os.path.join(script_dir, args.output_dir)
    else:
        output_base = args.output_dir
    
    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime('%m%d%Y_%H%M%S')
    
    if args.input:
        # Extract base name from input file for output naming
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        run_dir = os.path.join(output_base, f"{input_basename}_transcripts_{timestamp}")
    else:
        run_dir = os.path.join(output_base, f"transcript_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Output directory: {run_dir}\n")
    
    # Process single video or batch
    if args.video:
        # Single video mode
        results_df = pd.DataFrame([process_single_video(args.video, run_dir, args.language)])
        
    else:
        # Batch mode
        video_ids = read_input_file(args.input)
        
        if not video_ids:
            print("No video IDs found in input file!")
            return
        
        print(f"Processing {len(video_ids)} videos with {args.delay}s delay between requests\n")
        results_df = process_batch(video_ids, run_dir, args.language, args.delay)
    
    # Save master CSV with metadata
    master_csv = os.path.join(run_dir, 'transcript_metadata.csv')
    results_df.to_csv(master_csv, index=False)
    
    # Generate summary statistics
    total = len(results_df)
    successful = results_df['transcript_retrieved'].sum()
    failed = total - successful
    
    if successful > 0:
        avg_length = results_df[results_df['transcript_retrieved']]['transcript_length'].mean()
        total_words = results_df[results_df['transcript_retrieved']]['transcript_length'].sum()
    else:
        avg_length = 0
        total_words = 0
    
    # Save error log if there were failures
    if failed > 0:
        error_log = os.path.join(run_dir, 'errors.log')
        error_df = results_df[~results_df['transcript_retrieved']][['video_id', 'error_message']]
        error_df.to_csv(error_log, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print(f"\nTranscript statistics:")
        print(f"  Total words: {total_words:,}")
        print(f"  Average words per transcript: {avg_length:,.0f}")
    
    print(f"\nOutput files:")
    print(f"  Transcripts: {run_dir}/")
    print(f"  Metadata CSV: {master_csv}")
    if failed > 0:
        print(f"  Error log: {error_log}")
    
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    main()

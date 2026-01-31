#!/usr/bin/env python3
"""
YouTube Data Extraction Script
Usage: python youtube.py --channelname "NYFA"
"""

import argparse
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime
import os

# API Configuration
# NOTE: Do NOT hardcode API keys in this repo.
# Export an environment variable before running, e.g.:
#   export YT_API_KEY="..."
YT_API_KEY = os.getenv('YT_API_KEY')
if not YT_API_KEY:
    raise RuntimeError('Missing YT_API_KEY env var')

API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

def get_channel_id(youtube, channel_name):
    """
    Get channel ID from channel name/username.
    Tries multiple search methods.
    """
    # Try direct channel ID first (if user provides it)
    if channel_name.startswith('UC') and len(channel_name) == 24:
        return channel_name
    
    # Remove @ if present
    clean_name = channel_name.replace('@', '')
    
    # Search for channel
    request = youtube.search().list(
        part='snippet',
        q=clean_name,
        type='channel',
        maxResults=5
    )
    response = request.execute()
    
    if response['items']:
        # Return first match
        channel_id = response['items'][0]['snippet']['channelId']
        channel_title = response['items'][0]['snippet']['title']
        print(f"Found channel: {channel_title} ({channel_id})")
        return channel_id
    else:
        raise ValueError(f"Could not find channel: {channel_name}")

def get_channel_stats(youtube, channel_id):
    """Get channel statistics."""
    request = youtube.channels().list(
        part='statistics,snippet',
        id=channel_id
    )
    response = request.execute()
    
    if response['items']:
        item = response['items'][0]
        stats = item['statistics']
        snippet = item['snippet']
        
        return {
            'channel_name': snippet['title'],
            'subscribers': stats.get('subscriberCount', 0),
            'total_views': stats.get('viewCount', 0),
            'total_videos': stats.get('videoCount', 0)
        }
    return None

def get_all_videos(youtube, channel_id):
    """
    Get all videos from a channel.
    Returns list of video IDs.
    """
    # Get uploads playlist ID
    request = youtube.channels().list(
        part='contentDetails',
        id=channel_id
    )
    response = request.execute()
    
    if not response['items']:
        return []
    
    uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    
    # Get all videos from uploads playlist
    video_ids = []
    next_page_token = None
    
    while True:
        request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])
        
        next_page_token = response.get('nextPageToken')
        
        if not next_page_token:
            break
        
        print(f"Fetched {len(video_ids)} videos so far...")
    
    return video_ids

def get_video_details(youtube, video_ids):
    """
    Get detailed statistics for videos.
    YouTube API allows max 50 videos per request.
    """
    all_video_data = []
    
    # Process in batches of 50
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        
        request = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            id=','.join(batch)
        )
        response = request.execute()
        
        for item in response['items']:
            snippet = item['snippet']
            stats = item['statistics']
            details = item['contentDetails']
            
            # Parse duration from ISO 8601 format (PT1H2M3S)
            duration = details['duration']
            
            video_data = {
                'video_id': item['id'],
                'title': snippet['title'],
                'description': snippet['description'],
                'published_date': snippet['publishedAt'],
                'duration': duration,
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0)),
                'video_url': f"https://www.youtube.com/watch?v={item['id']}"
            }
            
            all_video_data.append(video_data)
        
        print(f"Processed {min(i+50, len(video_ids))}/{len(video_ids)} videos")
    
    return all_video_data

def parse_duration_to_seconds(duration):
    """Convert ISO 8601 duration to seconds."""
    import re
    
    # Remove PT prefix
    duration = duration.replace('PT', '')
    
    hours = 0
    minutes = 0
    seconds = 0
    
    # Extract hours
    if 'H' in duration:
        hours = int(re.search(r'(\d+)H', duration).group(1))
    
    # Extract minutes
    if 'M' in duration:
        minutes = int(re.search(r'(\d+)M', duration).group(1))
    
    # Extract seconds
    if 'S' in duration:
        seconds = int(re.search(r'(\d+)S', duration).group(1))
    
    return hours * 3600 + minutes * 60 + seconds

def calculate_derived_metrics(df):
    """Add calculated columns for analysis."""
    # Parse published date
    df['published_date'] = pd.to_datetime(df['published_date'])
    
    # Duration in seconds
    df['duration_seconds'] = df['duration'].apply(parse_duration_to_seconds)
    
    # Days since published - make timezone aware
    from datetime import timezone
    today = datetime.now(timezone.utc)
    df['days_since_published'] = (today - df['published_date']).dt.days
    df['days_since_published'] = df['days_since_published'].apply(lambda x: max(x, 1))  # Avoid division by zero
    
    # Engagement rate
    df['engagement_rate'] = ((df['like_count'] + df['comment_count']) / df['view_count'] * 100).round(2)
    df['engagement_rate'] = df['engagement_rate'].fillna(0)
    
    # Views per day
    df['views_per_day'] = (df['view_count'] / df['days_since_published']).round(2)
    
    # Likes per view
    df['likes_per_view'] = (df['like_count'] / df['view_count'] * 100).round(2)
    df['likes_per_view'] = df['likes_per_view'].fillna(0)
    
    return df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract YouTube channel data')
    parser.add_argument('--channelname', required=True, help='Channel name or ID')
    args = parser.parse_args()
    
    channel_name = args.channelname
    
    print(f"\n{'='*60}")
    print(f"YouTube Data Extraction for: {channel_name}")
    print(f"{'='*60}\n")
    
    # Initialize YouTube API
    youtube = build(API_SERVICE_NAME, API_VERSION, developerKey=YT_API_KEY)
    
    # Get channel ID
    print("Step 1: Finding channel...")
    channel_id = get_channel_id(youtube, channel_name)
    
    # Get channel stats
    print("\nStep 2: Getting channel statistics...")
    channel_stats = get_channel_stats(youtube, channel_id)
    if channel_stats:
        print(f"Channel: {channel_stats['channel_name']}")
        print(f"Subscribers: {int(channel_stats['subscribers']):,}")
        print(f"Total Views: {int(channel_stats['total_views']):,}")
        print(f"Total Videos: {int(channel_stats['total_videos']):,}")
    
    # Get all video IDs
    print("\nStep 3: Fetching all video IDs...")
    video_ids = get_all_videos(youtube, channel_id)
    print(f"Found {len(video_ids)} videos")
    
    if not video_ids:
        print("No videos found!")
        return
    
    # Get video details
    print("\nStep 4: Fetching video details...")
    video_data = get_video_details(youtube, video_ids)
    
    # Create DataFrame
    print("\nStep 5: Processing data...")
    df = pd.DataFrame(video_data)
    
    # Add channel info
    if channel_stats:
        df['channel_name'] = channel_stats['channel_name']
        df['channel_id'] = channel_id
    
    # Calculate derived metrics
    df = calculate_derived_metrics(df)
    
    # Sort by published date (newest first)
    df = df.sort_values('published_date', ascending=False)
    
    # Create output filename
    today = datetime.now()
    date_str = today.strftime('%m%d%Y')
    
    # Clean channel name for filename
    clean_channel_name = channel_name.replace('@', '').replace(' ', '_')
    output_filename = f"{clean_channel_name}_{date_str}.csv"
    
    # Ensure data directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, output_filename)
    
    # Export to CSV
    print(f"\nStep 6: Exporting to CSV...")
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"{'='*60}")
    print(f"Exported {len(df)} videos to:")
    print(f"{output_path}")
    print(f"\nColumns included:")
    print(f"  - video_id, title, description")
    print(f"  - published_date, duration, duration_seconds")
    print(f"  - view_count, like_count, comment_count")
    print(f"  - engagement_rate, views_per_day, likes_per_view")
    print(f"  - days_since_published, video_url")
    print(f"  - channel_name, channel_id")
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    main()

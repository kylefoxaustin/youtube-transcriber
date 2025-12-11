#!/usr/bin/env python3
"""
URL Extractor for YouTube
Extract video URLs from channels, playlists, or search results.
"""

import argparse
import sys
from datetime import datetime

try:
    import yt_dlp
except ImportError:
    print("yt-dlp not installed. Run: pip install yt-dlp")
    sys.exit(1)


def extract_urls(source: str, limit: int = None, sort_by: str = None) -> list:
    """Extract video URLs from a YouTube source."""
    
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    # Handle different sort options for channels
    if sort_by == 'oldest':
        # YouTube doesn't support this directly, we'll reverse after
        pass
    elif sort_by == 'popular':
        if '/videos' in source:
            source = source.replace('/videos', '/videos?view=0&sort=p')
    
    urls = []
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(source, download=False)
            
            if info is None:
                print(f"Could not extract info from: {source}")
                return []
            
            # Handle playlist/channel
            if 'entries' in info:
                entries = list(info['entries'])
                
                if sort_by == 'oldest':
                    entries = entries[::-1]
                
                for entry in entries:
                    if entry and entry.get('id'):
                        urls.append({
                            'url': f"https://www.youtube.com/watch?v={entry['id']}",
                            'id': entry['id'],
                            'title': entry.get('title', 'Unknown'),
                            'duration': entry.get('duration'),
                        })
                        
                        if limit and len(urls) >= limit:
                            break
            else:
                # Single video
                urls.append({
                    'url': f"https://www.youtube.com/watch?v={info['id']}",
                    'id': info['id'],
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration'),
                })
                
        except Exception as e:
            print(f"Error: {e}")
    
    return urls


def format_duration(seconds):
    """Format duration in human-readable form."""
    if not seconds:
        return "?"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract YouTube video URLs from channels, playlists, or search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from a channel (all videos)
  python extract_urls.py "https://www.youtube.com/@ChannelName/videos"

  # Extract from a playlist
  python extract_urls.py "https://www.youtube.com/playlist?list=PLxxxxxx"

  # Limit to first N videos
  python extract_urls.py --limit 50 "https://www.youtube.com/@ChannelName/videos"

  # Sort by oldest first
  python extract_urls.py --sort oldest "https://www.youtube.com/@ChannelName/videos"

  # Sort by most popular
  python extract_urls.py --sort popular "https://www.youtube.com/@ChannelName/videos"

  # Save to file
  python extract_urls.py "URL" --output urls.txt

  # Show details (titles, durations)
  python extract_urls.py --verbose "URL"
        """
    )
    
    parser.add_argument('source', help='YouTube URL (channel, playlist, or video)')
    parser.add_argument('--limit', '-n', type=int, help='Maximum number of URLs to extract')
    parser.add_argument('--sort', choices=['newest', 'oldest', 'popular'], default='newest',
                        help='Sort order (default: newest)')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show titles and durations')
    parser.add_argument('--format', choices=['plain', 'csv', 'markdown'], default='plain',
                        help='Output format (default: plain)')
    
    args = parser.parse_args()
    
    print(f"Extracting URLs from: {args.source}", file=sys.stderr)
    
    urls = extract_urls(args.source, limit=args.limit, sort_by=args.sort)
    
    if not urls:
        print("No videos found.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(urls)} videos", file=sys.stderr)
    
    # Format output
    output_lines = []
    
    if args.format == 'csv':
        output_lines.append("url,id,title,duration")
        for v in urls:
            title = v['title'].replace('"', '""')
            output_lines.append(f'"{v["url"]}","{v["id"]}","{title}","{v.get("duration", "")}"')
    
    elif args.format == 'markdown':
        output_lines.append("| # | Title | Duration | URL |")
        output_lines.append("|---|-------|----------|-----|")
        for i, v in enumerate(urls, 1):
            dur = format_duration(v.get('duration'))
            output_lines.append(f"| {i} | {v['title'][:50]} | {dur} | [Link]({v['url']}) |")
    
    else:  # plain
        for v in urls:
            if args.verbose:
                dur = format_duration(v.get('duration'))
                output_lines.append(f"# {v['title']} [{dur}]")
            output_lines.append(v['url'])
            if args.verbose:
                output_lines.append("")
    
    output_text = "\n".join(output_lines)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text + "\n")
        print(f"Saved to: {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Universal Video Downloader
Uses yt-dlp to download videos from various platforms legally.
Only use for content you have rights to or that's in the public domain.
"""

import sys
import os
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("yt-dlp is not installed. Install it with: pip install yt-dlp")
    sys.exit(1)


class VideoDownloader:
    def __init__(self, output_dir="downloads"):
        """Initialize the video downloader with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def download_video(self, url, quality="best", audio_only=False, format_type="mp4"):
        """
        Download a video from the given URL.
        
        Args:
            url: Video URL to download
            quality: Video quality ('best', 'worst', or specific height like '720', '1080')
            audio_only: If True, download only audio
            format_type: Output format (mp4, webm, mkv, etc.)
        """
        output_template = str(self.output_dir / '%(title)s.%(ext)s')
        
        if audio_only:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_template,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'progress_hooks': [self._progress_hook],
            }
        else:
            # Format selection based on quality
            if quality == "best":
                format_string = f'bestvideo[ext={format_type}]+bestaudio[ext=m4a]/best[ext={format_type}]/best'
            elif quality == "worst":
                format_string = 'worst'
            else:
                # Specific height (e.g., 720, 1080)
                format_string = f'bestvideo[height<={quality}][ext={format_type}]+bestaudio[ext=m4a]/best[height<={quality}]'
            
            ydl_opts = {
                'format': format_string,
                'outtmpl': output_template,
                'merge_output_format': format_type,
                'progress_hooks': [self._progress_hook],
            }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"\n📥 Downloading from: {url}")
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                print(f"\n✅ Downloaded successfully: {filename}")
                return filename
        except Exception as e:
            print(f"\n❌ Error downloading video: {e}")
            return None
    
    def download_playlist(self, url, quality="best", max_videos=None):
        """
        Download an entire playlist.
        
        Args:
            url: Playlist URL
            quality: Video quality
            max_videos: Maximum number of videos to download (None for all)
        """
        output_template = str(self.output_dir / '%(playlist)s/%(playlist_index)s - %(title)s.%(ext)s')
        
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': output_template,
            'progress_hooks': [self._progress_hook],
            'noplaylist': False,
        }
        
        if max_videos:
            ydl_opts['playlist_items'] = f'1-{max_videos}'
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"\n📥 Downloading playlist: {url}")
                ydl.download([url])
                print(f"\n✅ Playlist downloaded successfully")
        except Exception as e:
            print(f"\n❌ Error downloading playlist: {e}")
    
    def get_video_info(self, url):
        """Get information about a video without downloading."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                print(f"\n📹 Video Information:")
                print(f"Title: {info.get('title', 'N/A')}")
                print(f"Duration: {info.get('duration', 0)} seconds")
                print(f"Uploader: {info.get('uploader', 'N/A')}")
                print(f"Views: {info.get('view_count', 'N/A')}")
                print(f"Resolution: {info.get('width', 'N/A')}x{info.get('height', 'N/A')}")
                
                # Available formats
                if 'formats' in info:
                    print(f"\nAvailable formats: {len(info['formats'])}")
                    for fmt in info['formats'][:5]:  # Show first 5
                        print(f"  - {fmt.get('format_id')}: {fmt.get('format_note', 'N/A')} "
                              f"({fmt.get('ext', 'N/A')})")
                
                return info
        except Exception as e:
            print(f"\n❌ Error getting video info: {e}")
            return None
    
    def _progress_hook(self, d):
        """Display download progress."""
        if d['status'] == 'downloading':
            percent = d.get('_percent_str', 'N/A')
            speed = d.get('_speed_str', 'N/A')
            eta = d.get('_eta_str', 'N/A')
            print(f"\r⏳ Progress: {percent} | Speed: {speed} | ETA: {eta}", end='')
        elif d['status'] == 'finished':
            print(f"\n✓ Download finished, processing...")


def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal Video Downloader - Download videos from various platforms legally"
    )
    parser.add_argument("url", help="Video or playlist URL to download")
    parser.add_argument("-o", "--output", default="downloads", help="Output directory")
    parser.add_argument("-q", "--quality", default="best", 
                       help="Video quality: best, worst, 720, 1080, etc.")
    parser.add_argument("-a", "--audio-only", action="store_true", 
                       help="Download audio only (MP3)")
    parser.add_argument("-f", "--format", default="mp4", 
                       help="Output format: mp4, webm, mkv")
    parser.add_argument("-i", "--info", action="store_true", 
                       help="Show video info without downloading")
    parser.add_argument("-p", "--playlist", action="store_true", 
                       help="Download entire playlist")
    parser.add_argument("-m", "--max-videos", type=int, 
                       help="Maximum videos to download from playlist")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = VideoDownloader(output_dir=args.output)
    
    # Execute based on arguments
    if args.info:
        downloader.get_video_info(args.url)
    elif args.playlist:
        downloader.download_playlist(args.url, quality=args.quality, 
                                     max_videos=args.max_videos)
    else:
        downloader.download_video(args.url, quality=args.quality, 
                                 audio_only=args.audio_only, 
                                 format_type=args.format)


if __name__ == "__main__":
    # Example usage if run without arguments
    if len(sys.argv) == 1:
        print("Universal Video Downloader")
        print("=" * 50)
        print("\nUsage examples:")
        print("  python video_downloader.py <URL>")
        print("  python video_downloader.py <URL> -q 720")
        print("  python video_downloader.py <URL> -a  # audio only")
        print("  python video_downloader.py <URL> -i  # info only")
        print("  python video_downloader.py <URL> -p  # download playlist")
        print("\nInstall yt-dlp first: pip install yt-dlp")
        print("\nFor more options: python video_downloader.py --help")
        sys.exit(0)
    
    main()

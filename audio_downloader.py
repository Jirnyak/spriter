#!/usr/bin/env python3
"""
Batch Audio Downloader
Reads URLs from links.txt and downloads only audio from each video.
Removes each link from the file after successful download.
"""

import sys
import os
from pathlib import Path
from video_downloader import VideoDownloader

LINKS_FILE = "links.txt"


def read_links(filename):
    """Read all links from the file."""
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return []
    
    with open(filename, 'r', encoding='utf-8') as f:
        links = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    return links


def remove_link_from_file(filename, link_to_remove):
    """Remove a specific link from the file."""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip() != link_to_remove:
                f.write(line)


def batch_audio_download(links_file=LINKS_FILE, output_dir="audio_downloads", quality="best", audio_format="mp3"):
    """
    Download only audio from all videos in links file and remove each link after successful download.
    
    Args:
        links_file: Path to text file containing URLs (one per line)
        output_dir: Directory to save downloaded audio files
        quality: Audio quality setting (ignored, kept for interface)
        audio_format: Output audio format (mp3, m4a, etc.)
    """
    # Read all links
    links = read_links(links_file)
    
    if not links:
        print(f"⚠️  No links found in {links_file}")
        print(f"\nCreate {links_file} and add video URLs (one per line)")
        return
    
    print(f"📋 Found {len(links)} link(s) to download (audio only)")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔊 Audio format: {audio_format}")
    print("=" * 60)
    
    # Initialize downloader
    downloader = VideoDownloader(output_dir=output_dir)
    
    # Track statistics
    successful = 0
    failed = 0
    failed_links = []
    
    # Download each link
    for idx, link in enumerate(links, 1):
        print(f"\n\n{'=' * 60}")
        print(f"🔊 [{idx}/{len(links)}] Processing: {link}")
        print(f"{'=' * 60}")
        
        try:
            result = downloader.download_video(link, quality=quality, audio_only=True, format_type=audio_format)
            
            if result:
                # Successful download - remove link from file
                remove_link_from_file(links_file, link)
                successful += 1
                print(f"✅ Successfully downloaded audio and removed link from {links_file}")
            else:
                failed += 1
                failed_links.append(link)
                print(f"⚠️  Download failed but will try next link")
        
        except KeyboardInterrupt:
            print(f"\n\n⏸️  Download interrupted by user")
            print(f"✓ Processed: {successful + failed}/{len(links)}")
            print(f"✓ Successful: {successful}")
            print(f"✗ Failed: {failed}")
            print(f"\nRemaining links are still in {links_file}")
            sys.exit(0)
        
        except Exception as e:
            failed += 1
            failed_links.append(link)
            print(f"❌ Error: {e}")
            print(f"⚠️  Continuing to next link...")
    
    # Final summary
    print(f"\n\n{'=' * 60}")
    print(f"📊 AUDIO DOWNLOAD SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ Successful: {successful}/{len(links)}")
    print(f"❌ Failed: {failed}/{len(links)}")
    
    if failed_links:
        print(f"\n⚠️  Failed links (still in {links_file}):")
        for link in failed_links:
            print(f"  - {link}")
    else:
        print(f"\n🎉 All audio downloads completed successfully!")
        print(f"✓ {links_file} is now empty")


def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch Audio Downloader - Download only audio from links.txt"
    )
    parser.add_argument("-f", "--file", default=LINKS_FILE,
                       help=f"Links file (default: {LINKS_FILE})")
    parser.add_argument("-o", "--output", default="audio_downloads",
                       help="Output directory (default: audio_downloads)")
    parser.add_argument("-q", "--quality", default="best",
                       help="Audio quality (default: best)")
    parser.add_argument("-a", "--audio-format", default="mp3",
                       help="Audio format: mp3, m4a, etc.")
    
    args = parser.parse_args()
    
    batch_audio_download(
        links_file=args.file,
        output_dir=args.output,
        quality=args.quality,
        audio_format=args.audio_format
    )


if __name__ == "__main__":
    main()

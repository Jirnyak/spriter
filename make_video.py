#!/usr/bin/env python3
"""
Create MP4 video from iteration images.
"""

import os
import subprocess

def main():
    input_dir = "iterations"
    output_mp4 = "city_growth.mp4"
    output_webm = "city_growth.webm"
    
    # Check if iterations directory exists and has images
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} directory not found!")
        return
    
    images = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    if not images:
        print(f"Error: No PNG images found in {input_dir}!")
        return
    
    print(f"Found {len(images)} images")
    
    # Create MP4 video
    print(f"\nCreating MP4 video: {output_mp4}")
    cmd_mp4 = [
        'ffmpeg',
        '-y',
        '-framerate', '10',
        '-pattern_type', 'glob',
        '-i', f'{input_dir}/city_*.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '23',
        output_mp4
    ]
    
    try:
        result = subprocess.run(cmd_mp4, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ MP4 saved: {output_mp4}")
            size_mb = os.path.getsize(output_mp4) / (1024 * 1024)
            print(f"  File size: {size_mb:.2f} MB")
        else:
            print(f"Error creating MP4:")
            print(result.stderr)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg:")
        print("  brew install ffmpeg")
        return
    
    # Create WebM video (alternative format)
    print(f"\nCreating WebM video: {output_webm}")
    cmd_webm = [
        'ffmpeg',
        '-y',
        '-framerate', '10',
        '-pattern_type', 'glob',
        '-i', f'{input_dir}/city_*.png',
        '-c:v', 'libvpx-vp9',
        '-crf', '30',
        '-b:v', '0',
        output_webm
    ]
    
    try:
        result = subprocess.run(cmd_webm, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ WebM saved: {output_webm}")
            size_mb = os.path.getsize(output_webm) / (1024 * 1024)
            print(f"  File size: {size_mb:.2f} MB")
        else:
            print(f"Error creating WebM:")
            print(result.stderr)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

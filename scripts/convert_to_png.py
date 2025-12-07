#!/usr/bin/env python
"""Convert all images in a directory to PNG format."""

from PIL import Image
from pathlib import Path
import os
import sys

def convert_to_png(source_dir):
    """Convert all jpg, jpeg, and webp files to PNG."""
    source_dir = Path(source_dir)
    
    if not source_dir.exists():
        print(f"Error: Directory not found: {source_dir}")
        return
    
    converted = 0
    skipped = 0
    failed = 0
    
    for f in source_dir.iterdir():
        if f.suffix.lower() in ['.jpg', '.jpeg', '.webp']:
            try:
                img = Image.open(f).convert('RGB')
                new_path = f.with_suffix('.png')
                img.save(new_path, 'PNG')
                os.remove(f)  # Remove original
                print(f'✓ Converted: {f.name} -> {new_path.name}')
                converted += 1
            except Exception as e:
                print(f'✗ Failed: {f.name} - {e}')
                failed += 1
        elif f.suffix.lower() == '.png':
            skipped += 1
    
    print(f'\nDone! Converted {converted} files, skipped {skipped} existing PNGs, {failed} failed')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_to_png(sys.argv[1])
    else:
        print("Usage: python convert_to_png.py <directory>")

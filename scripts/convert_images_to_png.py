"""
Convert all images in a folder to PNG format.
Handles jpg, jpeg, webp, avif, bmp, gif, tiff and other formats.
"""
import os
import sys
from pathlib import Path
from PIL import Image

# Try to import pillow-avif for AVIF support
try:
    import pillow_avif
    AVIF_SUPPORT = True
except ImportError:
    AVIF_SUPPORT = False
    print("Note: pillow-avif not installed. AVIF files will be skipped.")
    print("Install with: pip install pillow-avif-plugin")

def convert_to_png(folder_path):
    """Convert all non-PNG images to PNG format."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return
    
    converted = 0
    skipped = 0
    errors = 0
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
    if AVIF_SUPPORT:
        image_extensions.add('.avif')
    
    for file_path in folder.glob('*'):
        if file_path.suffix.lower() in image_extensions:
            try:
                # Open and convert to RGB (in case of RGBA webp)
                img = Image.open(file_path).convert('RGB')
                
                # Save as PNG with same base name
                new_path = file_path.with_suffix('.png')
                
                # Handle name conflicts
                counter = 1
                while new_path.exists():
                    new_path = file_path.parent / f"{file_path.stem}_{counter}.png"
                    counter += 1
                
                img.save(new_path, 'PNG')
                
                print(f"✓ Converted: {file_path.name} → {new_path.name}")
                
                # Delete original
                file_path.unlink()
                
                converted += 1
                
            except Exception as e:
                print(f"✗ Error converting {file_path.name}: {e}")
                errors += 1
        
        elif file_path.suffix.lower() == '.png':
            skipped += 1
        
        elif file_path.suffix.lower() == '.avif' and not AVIF_SUPPORT:
            print(f"⚠ Skipped (no AVIF support): {file_path.name}")
            errors += 1
    
    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Converted: {converted}")
    print(f"  Already PNG: {skipped}")
    print(f"  Errors/Skipped: {errors}")


if __name__ == "__main__":
    # Check if a folder path was provided as argument
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        print(f"Processing folder: {folder}")
        print("="*50)
        convert_to_png(folder)
    else:
        # Default: process data/training_datasets
        datasets_dir = Path("data/training_datasets")
        
        if not datasets_dir.exists():
            print("Usage: python scripts/convert_images_to_png.py <folder_path>")
            print("Or run from project root to process data/training_datasets/")
            exit(1)
        
        for dataset_folder in datasets_dir.iterdir():
            if dataset_folder.is_dir():
                images_folder = dataset_folder / "images"
                if images_folder.exists():
                    print(f"\n{'='*50}")
                    print(f"Processing: {dataset_folder.name}")
                    print(f"{'='*50}")
                    convert_to_png(images_folder)

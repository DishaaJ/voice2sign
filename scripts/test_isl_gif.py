#!/usr/bin/env python3
"""
Test GIF generation using real ISL images
Creates an animated GIF showing sign language sequence
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io


def create_isl_gif(gloss_tokens: list, output_path: str, duration: int = 800):
    """Create GIF from ISL dataset images using gloss tokens"""
    data_path = Path('data')
    frames = []
    
    print(f"\n🎬 Creating GIF for: {' → '.join(gloss_tokens)}")
    
    for token in gloss_tokens:
        # Get first letter of token (LOVE → L, YOU → Y, ME → M)
        letter = token[0].upper()
        letter_folder = data_path / letter
        
        if not letter_folder.exists():
            print(f"   ⚠️  {token} ({letter}): folder not found")
            continue
        
        # Get first available image
        images = list(letter_folder.glob("*.jpg"))
        if not images:
            print(f"   ⚠️  {token} ({letter}): no images")
            continue
        
        # Load first image
        img = Image.open(images[0]).convert('RGB')
        print(f"   ✅ {token} ({letter}): loaded {images[0].name}")
        
        # Resize to consistent size
        img = img.resize((400, 400), Image.Resampling.LANCZOS)
        
        # Create frame with white background and label
        frame = Image.new('RGB', (500, 500), 'white')
        frame.paste(img, (50, 25))
        
        # Add label
        draw = ImageDraw.Draw(frame)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        except:
            font = ImageFont.load_default()
        
        # Draw token (full word)
        text_bbox = draw.textbbox((0, 0), token, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (500 - text_width) // 2
        draw.text((text_x, 450), token, fill='black', font=font)
        
        frames.append(frame)
    
    if not frames:
        print("❌ No frames created!")
        return
    
    # Create GIF
    print(f"\n💾 Saving GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    file_size = Path(output_path).stat().st_size / 1024
    print(f"✅ GIF created! Size: {file_size:.1f}KB")
    print(f"   Frames: {len(frames)}")
    print(f"   Duration per frame: {duration}ms")


if __name__ == "__main__":
    # Test with actual gloss tokens from "I love you"
    # The NLP converts "I love you" → ["LOVE"]
    # So we test with the full token words
    letters = ['LOVE', 'YOU', 'ME']  # Full gloss tokens
    
    output = 'output/isl_test.gif'
    create_isl_gif(letters, output, duration=1000)
    
    print(f"\n🎉 Test complete! View: {output}")

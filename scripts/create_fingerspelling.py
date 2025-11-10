#!/usr/bin/env python3
"""
Fingerspelling GIF Generator
Creates animated GIFs that spell out words letter by letter
E.g., "LOVE" → L → O → V → E animation
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io


def create_fingerspelling_gif(words: list, output_path: str, duration_per_letter: int = 300):
    """
    Create GIF showing fingerspelling of words
    
    Args:
        words: List of words to spell (e.g., ['LOVE', 'YOU', 'ME'])
        output_path: Path to save GIF
        duration_per_letter: Duration per letter frame in milliseconds (fast!)
    """
    data_path = Path('data')
    frames = []
    
    print(f"\n🎬 Creating Fingerspelling GIF")
    print(f"   Words: {' → '.join(words)}")
    print(f"   Speed: {duration_per_letter}ms per letter\n")
    
    for word_idx, word in enumerate(words, 1):
        print(f"📝 Word {word_idx}: {word}")
        
        letters = list(word.upper())
        word_frames = []
        
        for letter_idx, letter in enumerate(letters, 1):
            letter_folder = data_path / letter
            
            if not letter_folder.exists():
                print(f"   ⚠️  {letter}: folder not found, skipping")
                continue
            
            # Get first available image
            images = list(letter_folder.glob("*.jpg"))
            if not images:
                print(f"   ⚠️  {letter}: no images")
                continue
            
            # Load random image
            import random
            img = Image.open(random.choice(images)).convert('RGB')
            
            # Resize to consistent size
            img = img.resize((350, 350), Image.Resampling.LANCZOS)
            
            # Create frame with white background
            frame = Image.new('RGB', (500, 550), 'white')
            frame.paste(img, (75, 50))
            
            # Add letter label and progress
            draw = ImageDraw.Draw(frame)
            try:
                font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 72)
                font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
            except:
                font_big = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Draw current letter (large, centered at top)
            text_bbox = draw.textbbox((0, 0), letter, font=font_big)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (500 - text_width) // 2
            draw.text((text_x, 10), letter, fill='#2196F3', font=font_big)
            
            # Draw word being spelled
            text_bbox = draw.textbbox((0, 0), word, font=font_small)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (500 - text_width) // 2
            draw.text((text_x, 420), f"Spelling: {word}", fill='#333', font=font_small)
            
            # Draw progress (e.g., "2 of 4")
            progress_text = f"{letter_idx}/{len(letters)}"
            text_bbox = draw.textbbox((0, 0), progress_text, font=font_small)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (500 - text_width) // 2
            draw.text((text_x, 480), progress_text, fill='#999', font=font_small)
            
            word_frames.append(frame)
            print(f"   ✅ {letter} ({letter_idx}/{len(letters)})")
        
        # Add all letter frames to main frames list
        frames.extend(word_frames)
        
        # Add pause frame (same as last letter, but held longer)
        if word_frames:
            pause_frame = word_frames[-1].copy()
            frames.extend([pause_frame] * 2)  # Hold for 2 frames
            print(f"   ⏸️  Added pause between words\n")
    
    if not frames:
        print("❌ No frames created!")
        return
    
    # Create GIF
    print(f"💾 Saving GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_per_letter,
        loop=0,  # Loop forever
        optimize=False
    )
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    total_duration = (len(frames) * duration_per_letter) / 1000
    print(f"✅ GIF created!")
    print(f"   Size: {file_size:.2f}MB")
    print(f"   Frames: {len(frames)}")
    print(f"   Duration: {total_duration:.1f}s")
    print(f"   Speed: {duration_per_letter}ms per letter (FAST!)")


if __name__ == "__main__":
    # Test with "LOVE YOU"
    words = ['LOVE', 'YOU']
    
    output = 'output/fingerspelling_demo.gif'
    
    # Fast fingerspelling (300ms per letter = 3.3 letters per second)
    create_fingerspelling_gif(words, output, duration_per_letter=300)
    
    print(f"\n🎉 Test complete!")
    print(f"📺 View: {output}")
    print(f"\n📋 How to use in app:")
    print(f"   For each gloss word, generate fingerspelling GIF")
    print(f"   Show in animated sequence")
    print(f"   Let user see how to spell each word!")

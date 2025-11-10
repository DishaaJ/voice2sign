"""
Sign Language GIF Generator
Creates animated GIFs from transcripts showing sign language
Uses reference images and animations for better visual quality
"""
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
from typing import Optional, List, Tuple
from datetime import datetime


class SignLanguageGIFGenerator:
    """Generate animated GIFs for sign language sequences"""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.reference_img_path = dataset_path / "american_sign_language.PNG"
        self.reference_img = None
        self.sign_positions = self._map_sign_positions()
        self._load_reference()
    
    def _load_reference(self):
        """Load the reference sign language image"""
        if self.reference_img_path.exists():
            try:
                self.reference_img = Image.open(self.reference_img_path).convert('RGB')
                print(f"✅ Loaded reference image: {self.reference_img.size}")
            except Exception as e:
                print(f"❌ Error loading reference: {e}")
    
    def _map_sign_positions(self) -> dict:
        """
        Map letter positions in the reference image
        American Sign Language alphabet is typically arranged in grid
        """
        # Approximate positions for ASL alphabet (A-Z)
        # This is a rough mapping - adjust based on actual image layout
        positions = {
            'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0), 'E': (4, 0),
            'F': (0, 1), 'G': (1, 1), 'H': (2, 1), 'I': (3, 1), 'J': (4, 1),
            'K': (0, 2), 'L': (1, 2), 'M': (2, 2), 'N': (3, 2), 'O': (4, 2),
            'P': (0, 3), 'Q': (1, 3), 'R': (2, 3), 'S': (3, 3), 'T': (4, 3),
            'U': (0, 4), 'V': (1, 4), 'W': (2, 4), 'X': (3, 4), 'Y': (4, 4),
            'Z': (2, 5),
        }
        return positions
    
    def extract_sign_from_reference(self, letter: str) -> Optional[Image.Image]:
        """Extract a specific sign letter from the reference image with padding"""
        if not self.reference_img:
            return None
        
        if letter not in self.sign_positions:
            return None
        
        try:
            # Reference image is 876×489
            # Estimate grid: 5 columns × 6 rows
            col, row = self.sign_positions[letter]
            
            # Calculate crop box with padding to avoid cutting signs
            cell_width = self.reference_img.width // 5
            cell_height = self.reference_img.height // 6
            
            # Add 10% padding on each side
            padding_x = int(cell_width * 0.05)
            padding_y = int(cell_height * 0.05)
            
            left = max(0, col * cell_width - padding_x)
            top = max(0, row * cell_height - padding_y)
            right = min(self.reference_img.width, (col + 1) * cell_width + padding_x)
            bottom = min(self.reference_img.height, (row + 1) * cell_height + padding_y)
            
            # Crop with padding
            cropped = self.reference_img.crop((left, top, right, bottom))
            
            # Resize to 400×400 for display
            resized = cropped.resize((400, 400), Image.Resampling.LANCZOS)
            
            return resized
        except Exception as e:
            print(f"Error extracting {letter}: {e}")
            return None
    
    def create_sign_sequence_gif(self, gloss_tokens: List[str], duration: int = 500) -> Optional[bytes]:
        """
        Create an animated GIF showing a sequence of signs
        
        Args:
            gloss_tokens: List of sign tokens (e.g., ["LOVE", "YOU", "ME"])
            duration: Duration per frame in milliseconds
        
        Returns:
            GIF bytes or None
        """
        frames = []
        
        # Create a frame for each sign
        for token in gloss_tokens:
            # Get first letter of token
            letter = token[0].upper() if token else '?'
            
            # Extract sign from reference
            sign_img = self.extract_sign_from_reference(letter)
            
            if sign_img:
                # Create larger frame to prevent cutting (550×600)
                frame = Image.new('RGB', (550, 600), 'white')
                
                # Paste sign image centered with more space
                x_offset = (550 - 400) // 2  # Center horizontally
                y_offset = 50  # Space from top
                frame.paste(sign_img, (x_offset, y_offset))
                
                # Add text label at bottom
                draw = ImageDraw.Draw(frame)
                try:
                    # Try to use a nice font
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
                except:
                    font = ImageFont.load_default()
                
                # Draw token name at bottom
                text_bbox = draw.textbbox((0, 0), token, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = (550 - text_width) // 2
                draw.text((text_x, 480), token, fill='black', font=font)
                
                frames.append(frame)
            else:
                # Fallback frame with just text
                frame = Image.new('RGB', (550, 600), '#f0f0f0')
                draw = ImageDraw.Draw(frame)
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
                except:
                    font = ImageFont.load_default()
                
                # Draw large letter centered
                text_bbox = draw.textbbox((0, 0), letter, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = (550 - text_width) // 2
                text_y = (250 - text_height) // 2
                draw.text((text_x, text_y), letter, fill='#666', font=font)
                
                # Draw token below
                try:
                    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
                except:
                    font_small = ImageFont.load_default()
                text_bbox = draw.textbbox((0, 0), token, font=font_small)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = (550 - text_width) // 2
                draw.text((text_x, 400), token, fill='black', font=font_small)
                
                frames.append(frame)
        
        if not frames:
            return None
        
        # Create GIF from frames
        try:
            gif_buffer = io.BytesIO()
            frames[0].save(
                gif_buffer,
                format='GIF',
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,  # Loop forever
                optimize=False
            )
            gif_buffer.seek(0)
            return gif_buffer.getvalue()
        except Exception as e:
            print(f"Error creating GIF: {e}")
            return None


def get_gif_generator(dataset_path: Path) -> SignLanguageGIFGenerator:
    """Get or create the global GIF generator"""
    global _gif_generator
    if '_gif_generator' not in globals():
        _gif_generator = SignLanguageGIFGenerator(dataset_path)
    return _gif_generator

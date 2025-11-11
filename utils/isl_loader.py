"""
Indian Sign Language Image Loader
Uses the ISL dataset with real hand sign images
Provides high-quality sign language visuals
"""
import os
from pathlib import Path
from PIL import Image
from typing import Optional, List
import random


class ISLImageLoader:
    """Load images from Indian Sign Language dataset"""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
        self.image_cache = {}
        print(f"[🤟] Initializing ISL Image Loader from {dataset_path}")
        self._check_available_letters()
    
    def _check_available_letters(self):
        """Check which letter folders exist"""
        self.available = {}
        for letter in self.letters:
            letter_path = self.dataset_path / letter
            if letter_path.exists():
                # Count images
                images = list(letter_path.glob("*.jpg")) + list(letter_path.glob("*.png"))
                self.available[letter] = len(images)
                if len(images) > 0:
                    print(f"   ✅ {letter}: {len(images)} images")
            else:
                self.available[letter] = 0
    
    def get_sign_image(self, letter: str, size: int = 400) -> Optional[Image.Image]:
        """
        Get a random sign image for a letter or word
        
        Args:
            letter: Single letter (A-Z), digit (1-9), or full word (LOVE, HELLO, etc)
            size: Output image size (default 400×400)
        
        Returns:
            PIL Image or None
        """
        try:
            letter = letter.upper()
            
            # If it's a word, take first letter
            if len(letter) > 1:
                # For words like "LOVE", use first letter "L"
                letter = letter[0]
            
            # Check if letter exists
            if letter not in self.available or self.available[letter] == 0:
                return None
            
            # Get letter folder
            letter_path = self.dataset_path / letter
            
            # Get all images
            images = list(letter_path.glob("*.jpg")) + list(letter_path.glob("*.png"))
            if not images:
                return None
            
            # Pick random image
            img_path = random.choice(images)
            
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Resize to requested size
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            
            return img
        except Exception as e:
            print(f"❌ Error loading image for {letter}: {e}")
            return None
    
    def get_sign_batch(self, letter: str, num_samples: int = 3, size: int = 400) -> List[Image.Image]:
        """
        Get multiple sign images for a letter
        
        Args:
            letter: Single letter (A-Z) or digit (1-9)
            num_samples: Number of samples to return
            size: Image size
        
        Returns:
            List of PIL Images
        """
        try:
            letter = letter.upper()
            
            if letter not in self.available or self.available[letter] == 0:
                return []
            
            letter_path = self.dataset_path / letter
            images_list = list(letter_path.glob("*.jpg")) + list(letter_path.glob("*.png"))
            
            if not images_list:
                return []
            
            # Sample random images
            samples = random.sample(images_list, min(num_samples, len(images_list)))
            
            # Load and resize
            result = []
            for img_path in samples:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((size, size), Image.Resampling.LANCZOS)
                    result.append(img)
                except:
                    pass
            
            return result
        except Exception as e:
            print(f"❌ Error loading batch for {letter}: {e}")
            return []


# Global loader instance
_isl_loader = None


def get_isl_loader(dataset_path: Path) -> ISLImageLoader:
    """Get or create the global ISL image loader"""
    global _isl_loader
    if _isl_loader is None:
        _isl_loader = ISLImageLoader(dataset_path)
    return _isl_loader

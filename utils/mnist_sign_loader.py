"""
Sign Language MNIST Data Handler with Reference Images
Converts MNIST dataset to visual sign language representations
Uses high-quality reference images for better clarity
"""
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
from PIL import Image
import io
from typing import Optional, Dict, List


# MNIST Sign classes (A-Z, excluding J)
SIGN_CLASSES = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'Z'
}

LABEL_TO_SIGN = {v: k for k, v in SIGN_CLASSES.items()}


class MNISTSignLoader:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.train_csv = dataset_path / "sign_mnist_train.csv"
        self.test_csv = dataset_path / "sign_mnist_test.csv"
        
        # Reference images
        self.amer_sign2 = dataset_path / "amer_sign2.png"
        self.amer_sign3 = dataset_path / "amer_sign3.png"
        self.american_sign_language = dataset_path / "american_sign_language.PNG"
        
        self.data_cache = {}
        self.train_df = None
        self.test_df = None
        self._load_data()
    
    def _load_data(self):
        """Load MNIST CSV data once"""
        try:
            if self.train_csv.exists():
                print("[📊] Loading MNIST train data...")
                self.train_df = pd.read_csv(self.train_csv)
                print(f"✅ Loaded {len(self.train_df)} training samples")
            
            if self.test_csv.exists():
                print("[📊] Loading MNIST test data...")
                self.test_df = pd.read_csv(self.test_csv)
                print(f"✅ Loaded {len(self.test_df)} test samples")
        except Exception as e:
            print(f"❌ Error loading MNIST data: {e}")
            self.train_df = None
            self.test_df = None
    
    def get_reference_image(self) -> Optional[Image.Image]:
        """Get the best available reference image"""
        # Try in order of preference
        for img_path in [self.american_sign_language, self.amer_sign2, self.amer_sign3]:
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    return img.convert('RGB')
                except:
                    continue
        return None
    
    def get_sign_image(self, class_id, size: int = 300) -> Optional[Image.Image]:
        """
        Get a sign image for a specific class - MNIST data upscaled to high quality
        
        Args:
            class_id: Letter (str) or index (int) 
            size: Output image size (default 300x300)
        
        Returns:
            PIL Image of the sign at specified size
        """
        try:
            # Handle both letter (str) and index (int)
            if isinstance(class_id, str):
                class_id = LABEL_TO_SIGN.get(class_id.upper(), None)
                if class_id is None:
                    return None
            
            # Use train data if available
            df = self.train_df if self.train_df is not None else self.test_df
            if df is None:
                return None
            
            # Get samples for this class
            samples = df[df['label'] == class_id]
            if samples.empty:
                return None
            
            # Pick random sample
            sample = samples.sample(1).iloc[0]
            img_data = sample.drop('label').values.astype(np.uint8)
            
            # Reshape to 28x28
            img = Image.fromarray(img_data.reshape(28, 28), mode='L')
            
            # Upscale to specified size with high-quality resampling
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            
            # Enhance contrast for better visibility
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Increase contrast by 50%
            
            # Convert to RGB with white background
            img_rgb = Image.new('RGB', img.size, 'white')
            img_rgb.paste(img)
            
            return img_rgb
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def get_sign_batch(self, sign_label: str, num_samples=3):
        """Get multiple sample images for a sign"""
        if sign_label not in LABEL_TO_SIGN:
            return []
        
        label_idx = LABEL_TO_SIGN[sign_label]
        df = self.train_df if self.train_df is not None else self.test_df
        
        if df is None:
            return []
        
        samples = df[df['label'] == label_idx].head(num_samples)
        images = []
        
        for idx, row in samples.iterrows():
            pixels = row.drop('label').values.astype(np.uint8)
            image = pixels.reshape(28, 28)
            pil_image = Image.fromarray(image)
            images.append(pil_image)
        
        return images


def create_sign_grid_image(sign_label: str, mnist_loader: MNISTSignLoader, num_samples=3):
    """
    Create a grid of sign samples for display
    Returns: PIL Image showing multiple samples of the sign
    """
    images = mnist_loader.get_sign_batch(sign_label, num_samples)
    
    if not images:
        return None
    
    # Create grid image
    grid_size = 28
    num_cols = min(num_samples, len(images))
    num_rows = 1
    
    grid_image = Image.new('L', (grid_size * num_cols + 5 * (num_cols - 1), grid_size * num_rows), color=255)
    
    for col, img in enumerate(images):
        x_offset = col * (grid_size + 5)
        grid_image.paste(img, (x_offset, 0))
    
    return grid_image


def display_sign_in_streamlit(sign_label: str, mnist_loader: MNISTSignLoader, width=150):
    """Display a sign with label in Streamlit - LARGER"""
    image = mnist_loader.get_sign_image(sign_label)
    
    if image:
        # Create a nice display with label
        st.image(image, caption=f"🤟 {sign_label}", width=width, use_container_width=False)
    else:
        # Fallback
        st.markdown(f"""
        <div style="border: 2px dashed #999; border-radius: 8px; padding: 15px; text-align: center; background: #f5f5f5;">
            <p style="font-size: 28px; font-weight: bold; color: #666; margin: 0;">{sign_label}</p>
            <p style="font-size: 11px; color: #999; margin: 5px 0 0 0;">No MNIST data</p>
        </div>
        """, unsafe_allow_html=True)


# Global loader instance
_mnist_loader = None

def get_mnist_loader(dataset_path: Path) -> MNISTSignLoader:
    """Get or create the global MNIST loader"""
    global _mnist_loader
    if _mnist_loader is None:
        _mnist_loader = MNISTSignLoader(dataset_path)
    return _mnist_loader


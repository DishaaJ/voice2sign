#!/usr/bin/env python3
"""
Dual GIF Generator - Letter-wise and Token-wise animations
Uses existing create_fingerspelling utility
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from scripts.create_fingerspelling import create_fingerspelling_gif


class DualGIFGenerator:
    """Generate both letter-wise and token-wise GIFs using fingerspelling utility"""
    
    def __init__(self, output_path: str = "output"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
    
    def create_letter_gif(self, letter: str) -> str:
        """
        Create individual letter GIF using fingerspelling utility
        
        Args:
            letter: Single letter (A-Z, 1-9)
            
        Returns:
            Path to generated GIF
        """
        letter = letter.upper()
        
        print(f"📸 Creating letter GIF: {letter}")
        
        # Use fingerspelling for single letter
        output_file = self.output_path / f"letter_{letter.lower()}.gif"
        
        try:
            create_fingerspelling_gif([letter], str(output_file), duration_per_letter=800)
            print(f"   ✅ Saved: {output_file.name}")
            return str(output_file)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def create_token_gif(self, token: str) -> str:
        """
        Create word/token GIF using fingerspelling utility
        
        Args:
            token: Word to spell (e.g., "LOVE")
            
        Returns:
            Path to generated GIF
        """
        token = token.upper()
        
        print(f"📝 Creating token GIF: {token}")
        
        # Use fingerspelling for full word (300ms per letter = fast)
        output_file = self.output_path / f"token_{token.lower()}.gif"
        
        try:
            create_fingerspelling_gif([token], str(output_file), duration_per_letter=300)
            print(f"   ✅ Saved: {output_file.name}")
            return str(output_file)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def create_gloss_tokens_gif(self, gloss_tokens: list) -> str:
        """
        Create GIF for full gloss token sequence
        
        Args:
            gloss_tokens: List of tokens (e.g., ['LOVE', 'YOU'])
            
        Returns:
            Path to generated GIF
        """
        gloss_clean = [g for g in gloss_tokens if g != '|']
        
        print(f"🎬 Creating full gloss GIF: {' → '.join(gloss_clean)}")
        
        output_file = self.output_path / f"gloss_full.gif"
        
        try:
            create_fingerspelling_gif(gloss_clean, str(output_file), duration_per_letter=250)
            print(f"   ✅ Saved: {output_file.name}")
            return str(output_file)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None


# Global instance
_generator = None

def get_dual_gif_generator(output_path: str = "output"):
    """Get or create global generator instance"""
    global _generator
    if _generator is None:
        _generator = DualGIFGenerator(output_path)
    return _generator


if __name__ == "__main__":
    gen = DualGIFGenerator()
    
    print("=" * 60)
    print("🎬 DUAL GIF GENERATOR")
    print("=" * 60)
    
    # Test letter-wise
    print("\n📝 Test 1: Individual Letters")
    print("-" * 60)
    gen.create_letter_gif("L")
    gen.create_letter_gif("O")
    gen.create_letter_gif("V")
    gen.create_letter_gif("E")
    
    # Test token-wise
    print("\n📝 Test 2: Full Word Tokens")
    print("-" * 60)
    gen.create_token_gif("LOVE")
    gen.create_token_gif("YOU")
    gen.create_token_gif("ME")
    
    # Test gloss sequence
    print("\n📝 Test 3: Full Gloss Sequence")
    print("-" * 60)
    gen.create_gloss_tokens_gif(['LOVE', 'YOU', 'ME'])
    
    print("\n" + "=" * 60)
    print("✨ GIF Generation Complete!")
    print("=" * 60)
    print(f"\n🚀 Ready for frontend integration!")

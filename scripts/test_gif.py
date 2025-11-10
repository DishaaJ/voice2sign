from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.gif_generator import SignLanguageGIFGenerator

OUT = Path('output')
OUT.mkdir(exist_ok=True)

dataset = Path('datasets/sign-language-mnist')
print('Dataset exists:', dataset.exists())

gen = SignLanguageGIFGenerator(dataset)
print('Reference loaded:', gen.reference_img is not None)

tokens = ['I', 'LOVE', 'YOU']
print('Creating GIF for tokens:', tokens)

gif_bytes = gen.create_sign_sequence_gif(tokens, duration=600)
if gif_bytes:
    out_file = OUT / 'test_sign.gif'
    with open(out_file, 'wb') as f:
        f.write(gif_bytes)
    print('Wrote GIF to', out_file)
else:
    print('Failed to create GIF')

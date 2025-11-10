# 🤟 Voice2Sign - YouTube to Sign Language

Convert YouTube videos to **Sign Language Learning** with animated fingerspelling and real ISL images.

## ✨ Features

### 🎬 What It Does
1. **Download** YouTube video → Extract audio
2. **Transcribe** Audio → English text (Whisper AI)
3. **Convert** Text → Sign language gloss tokens (spaCy NLP)
4. **Detect** Emotion in text (DistilBERT)
5. **Visualize** Sign language with:
   - 🎬 **Animated Fingerspelling** (L→O→V→E)
   - 📸 **Real ISL Images** (42,000 hand photos)
   - 🎭 **Emotion Badges** (joy, sadness, etc.)

### ⚡ Performance
- **First run:** ~60 seconds (download, transcribe, process)
- **Cached repeat:** <1 second (instant loading)

### 🎨 Visual Learning
- Real Indian Sign Language (ISL) images (1200 per letter)
- Fingerspelling animations at learner-friendly speed
- Full word context (not fragmented tokens)
- Emotion context markers

## 🚀 Quick Start

### Local Development

```bash
# Clone & setup
git clone https://github.com/dk-a-dev/voice2sign.git
cd voice2sign

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run app
streamlit run app.py
```

Then open http://localhost:8501

### Docker

```bash
# Build & run
docker-compose up --build

# Access at http://localhost:8501
```

## 📝 Usage

1. **Paste YouTube URL**
   ```
   https://www.youtube.com/watch?v=dQw4w9WgXcQ
   ```

2. **Click "▶️ Analyze"**
   - Downloads audio
   - Transcribes with Whisper
   - Converts to sign language gloss
   - Detects emotion
   - Builds timeline

3. **View Results**
   - 📄 English transcription
   - 🎬 Animated fingerspelling for each word
   - 📸 Real ISL images
   - 🎭 Emotion indicators
   - ⏱️ Timestamp alignment

4. **Download**
   - 📋 Full transcript (TXT)
   - 🤟 Sign timeline (JSON)

## 📁 Project Structure

```
voice2sign/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── Dockerfile               # Docker config
├── docker-compose.yml       # Docker compose
│
├── modules/                 # Pipeline stages
│   ├── stage1_youtube.py    # Download audio
│   ├── stage1_transcribe.py # Whisper transcription
│   ├── stage2_nlp.py        # NLP gloss conversion
│   ├── stage2_emotion.py    # Emotion detection
│   └── stage3_map.py        # Timeline building
│
├── utils/                   # Utilities
│   ├── config.py           # Configuration
│   ├── cache_manager.py    # Per-stage caching
│   └── isl_loader.py       # ISL image loading
│
├── scripts/                # Helper scripts
│   └── create_fingerspelling.py  # Generate fingerspelling GIFs
│
├── data/                   # ISL Dataset (42,000 images)
├── output/                 # Generated files
└── cache/                  # Video cache
```

## 🎯 Pipeline Stages

### Stage 1: Download YouTube Audio
- Extract audio from YouTube URL
- Convert to WAV format
- Cache for repeat runs

### Stage 1b: Transcribe with Whisper
- OpenAI Whisper speech-to-text
- Multiple quality levels (tiny to large)
- Timestamps included
- Cache results

### Stage 2: NLP Gloss Conversion
- spaCy lemmatization
- Extract meaningful tokens
- Remove stopwords
- Convert to sign language gloss tokens
- Example: "I love you" → `LOVE YOU`

### Stage 2b: Emotion Detection
- DistilBERT text classification
- Emotions: joy, sadness, anger, fear, neutral
- Confidence score (0-100%)
- Associates with text segments

### Stage 3: Timeline Building
- Align sign tokens with timestamps
- Build temporal sequence
- Create downloadable timeline
- Support for multiple speed settings

## 🎬 Fingerspelling Animation

### How It Works
```
Input:  gloss_tokens = ['LOVE', 'YOU']
        duration_per_letter = 300ms

Process:
  - Load real ISL images for each letter
  - Create frame for each letter
  - Add letter label and progress counter
  - Animate at 300ms per letter

Output: Animated GIF
  L(0ms) → O(300ms) → V(600ms) → E(900ms) [pause] →
  Y(1400ms) → O(1700ms) → U(2000ms)
```

### Example Output
```
🎬 Creating Fingerspelling GIF
   Words: LOVE → YOU
   Speed: 300ms per letter

📝 Word 1: LOVE
   ✅ L (1/4)
   ✅ O (2/4)
   ✅ V (3/4)
   ✅ E (4/4)
   ⏸️  Added pause between words

✅ GIF created!
   Size: 0.48MB
   Frames: 11
   Duration: 3.3s
```

## 💾 Caching System

### Per-Stage Caching
Each video gets 5 cache files:

```
cache/
├── video_<YOUTUBE_ID>_stage_download.json      # Audio file info
├── video_<YOUTUBE_ID>_stage_transcribe.json    # Transcribed text
├── video_<YOUTUBE_ID>_stage_gloss.json         # Gloss tokens
├── video_<YOUTUBE_ID>_stage_emotion.json       # Emotion labels
└── video_<YOUTUBE_ID>_stage_timeline.json      # Final timeline
```

### Performance
- **Without cache:** 60-120 seconds
- **With cache:** <1 second
- Caching is automatic (transparent to user)

## 📊 Supported Languages

Currently: **English** (with Whisper's language detection)

Can be extended for:
- Hindi, Tamil, Telugu, Kannada, Malayalam (Indian languages)
- Other Indian Sign Language variants

## 📦 Dependencies

Core:
- `streamlit` - Web UI
- `whisper` - Speech-to-text
- `spacy` - NLP
- `torch` - Deep learning framework
- `transformers` - DistilBERT for emotion
- `yt-dlp` - YouTube download
- `Pillow` - Image processing
- `ffmpeg` - Audio codec

See `requirements.txt` for full list with versions.

## 🐳 Docker Support

### Build
```bash
docker-compose build
```

### Run
```bash
docker-compose up
```

### Volumes
- `./output` - Generated files
- `./cache` - Video cache
- `./data` - ISL dataset (required)

## 🧹 Cleanup & Optimization

### Remove Old Files
```bash
chmod +x cleanup.sh
./cleanup.sh
```

This removes:
- Old MNIST dataset
- Avatar training files
- Unnecessary documentation

## 📄 Configuration

### `.env` File
```bash
# Whisper model size
WHISPER_MODEL=base

# Cache directory
CACHE_DIR=./cache

# Output directory
OUTPUT_DIR=./output

# Emotion detection
ENABLE_EMOTION=true
```

## 🎓 Sign Language Learning

### What Users Learn
1. **Fingerspelling** - How to spell words letter by letter
2. **Sign tokens** - Main concepts for each word
3. **Emotion context** - How emotion affects signing
4. **Visual learning** - Real hand sign images

### ISL Dataset (42,000 Images)
- 1200 images per letter (A-Z)
- 1200 images per digit (1-9)
- High-resolution color photos
- Real human hands

## 🤝 Contributing

Contributions welcome! Areas to improve:
- Add more Indian Sign Language variants
- Improve gloss conversion
- Add video examples for emotions
- Enhance fingerspelling animations
- Support more languages

## 📄 License

MIT License - See LICENSE file

## 🙏 Credits

- **Whisper** - OpenAI speech-to-text
- **spaCy** - NLP library
- **DistilBERT** - Hugging Face emotion detection
- **ISL Dataset** - Real sign language photos
- **Streamlit** - Web framework
- **yt-dlp** - YouTube downloader

## ❓ FAQ

### Q: Do I need the ISL dataset?
**A:** Yes, download the Indian Sign Language dataset (42k images) and place in `/data/` folder.

### Q: Can I use this for other languages?
**A:** Yes, Whisper supports 99+ languages. Gloss conversion needs language-specific NLP.

### Q: How long does processing take?
**A:** First run ~60s (includes download & transcription). Cached repeat runs <1s.

### Q: Can I adjust animation speed?
**A:** Yes, modify `duration_per_letter` parameter in `create_fingerspelling.py`:
```python
# Fast (200ms)
create_fingerspelling_gif(words, output, duration_per_letter=200)

# Slow (500ms)
create_fingerspelling_gif(words, output, duration_per_letter=500)
```

### Q: How do I contribute a YouTube video example?
**A:** Create a PR with video URL in TEST_VIDEOS.md

## 🚀 Deployment

### Local
```bash
streamlit run app.py
```

### Docker (Recommended)
```bash
docker-compose up
```

### Cloud (Heroku, AWS, etc.)
- Use `Dockerfile`
- Mount `/data/` volume for ISL dataset
- Set environment variables

## 📞 Support

For issues, questions, or suggestions:
- GitHub Issues: Create issue on repository
- Email: devkeshwani@example.com
- Twitter: @dk_a_dev

---

**Made with ❤️ for Sign Language Learning**

🤟 Learn. Practice. Communicate. 🤟

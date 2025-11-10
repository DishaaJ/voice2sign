"""
🤟 Voice2Sign: YouTube Video → Captions + Sign Language
Complete app with real ISL (Indian Sign Language) images
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import json
import re
from utils.config import Config
from utils.helpers import read_json, write_json
from utils.isl_loader import get_isl_loader
from utils.cache_manager import CacheManager
from modules.stage1_youtube import download_youtube_audio
from modules.stage1_transcribe import transcribe_wav
from modules.stage2_nlp import process_segments_to_gloss, save_gloss_json
from modules.stage2_emotion import add_emotion_to_segments
from modules.stage3_map import build_sign_timeline, save_timeline_json
from scripts.create_fingerspelling import create_fingerspelling_gif
from scripts.dual_gif_generator import get_dual_gif_generator
import uuid


# ============================================================
# HELPER: Extract YouTube video ID
# ============================================================
def extract_youtube_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Voice2Sign",
    page_icon="🤟",
    layout="wide",
)

st.title("Voice2Sign")
st.markdown("**YouTube → Transcription → Sign Language with MNIST Visualization**")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    whisper_model = st.selectbox(
        "Transcription Model",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="tiny=fastest, large=most accurate"
    )
    
    show_emotion = st.checkbox("Include Emotion Analysis", value=False)
    
    st.divider()
    
    # Cache info
    cache_mgr = CacheManager()
    cached_videos = cache_mgr.list_cached_videos()
    if cached_videos:
        st.write(f"**💾 Cached Videos:** {len(cached_videos)}")
        if st.button("🗑️ Clear Cache"):
            cache_mgr.clear_cache()
            st.success("Cache cleared!")
            st.rerun()


# ============================================================
# INPUT
# ============================================================
st.header("📺 Input")

youtube_url = st.text_input(
    "YouTube URL:",
    placeholder="https://www.youtube.com/watch?v=...",
)

col1, col2 = st.columns([2, 1])
with col1:
    process_btn = st.button("▶️ Analyze", type="primary", use_container_width=True)
with col2:
    if youtube_url:
        video_id = extract_youtube_id(youtube_url)
        if video_id:
            cache_mgr = CacheManager()
            if cache_mgr.has_cache(video_id):
                st.success("✅ Cached")


# ============================================================
# PROCESS
# ============================================================
if process_btn and youtube_url:
    try:
        video_id = extract_youtube_id(youtube_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL")
            st.stop()
        
        st.session_state.youtube_url = youtube_url
        cfg = Config()
        cfg.whisper_model = whisper_model
        cfg.ensure_dirs()
        
        cache_mgr = CacheManager()
        stages_from_cache = []
        
        # ===== STAGE 1: DOWNLOAD =====
        stage1_cached = cache_mgr.load_stage_cache(video_id, "download")
        if stage1_cached:
            st.info("⚡ Using cached audio download")
            wav_path = Path(stage1_cached["wav_path"])
            stages_from_cache.append("download")
        else:
            with st.spinner("⏳ Downloading audio..."):
                wav_path = download_youtube_audio(youtube_url, cfg)
            st.success("✅ Downloaded")
            cache_mgr.save_stage_cache(video_id, "download", {"wav_path": str(wav_path)})
        
        # ===== STAGE 1: TRANSCRIBE =====
        stage1_transcribe_cached = cache_mgr.load_stage_cache(video_id, "transcribe")
        if stage1_transcribe_cached:
            st.info("⚡ Using cached transcription")
            transcript_json = stage1_transcribe_cached
            stages_from_cache.append("transcribe")
        else:
            with st.spinner("🎤 Transcribing..."):
                transcript_json = transcribe_wav(wav_path, cfg)
            st.success(f"✅ Got {len(transcript_json['segments'])} segments")
            cache_mgr.save_stage_cache(video_id, "transcribe", transcript_json)
        
        # ===== STAGE 2a: NLP (GLOSS) =====
        stage2_nlp_cached = cache_mgr.load_stage_cache(video_id, "gloss")
        if stage2_nlp_cached:
            st.info("⚡ Using cached gloss conversion")
            gloss_json = stage2_nlp_cached
            stages_from_cache.append("gloss")
        else:
            with st.spinner("🔠 Converting to gloss..."):
                gloss_json = process_segments_to_gloss(transcript_json["segments"], cfg)
            st.success("✅ Gloss ready")
            cache_mgr.save_stage_cache(video_id, "gloss", gloss_json)
        
        # ===== STAGE 2b: EMOTION =====
        if show_emotion:
            stage2_emotion_cached = cache_mgr.load_stage_cache(video_id, "emotion")
            if stage2_emotion_cached:
                st.info("⚡ Using cached emotion analysis")
                gloss_with_emotion = stage2_emotion_cached
                stages_from_cache.append("emotion")
            else:
                with st.spinner("🎭 Analyzing emotion..."):
                    gloss_with_emotion = add_emotion_to_segments(gloss_json, cfg)
                st.success("✅ Emotion analyzed")
                cache_mgr.save_stage_cache(video_id, "emotion", gloss_with_emotion)
        else:
            gloss_with_emotion = gloss_json
        
        # ===== STAGE 3: TIMELINE =====
        stage3_cached = cache_mgr.load_stage_cache(video_id, "timeline")
        if stage3_cached:
            st.info("⚡ Using cached timeline")
            timeline = stage3_cached
            stages_from_cache.append("timeline")
        else:
            with st.spinner("🤟 Building timeline..."):
                timeline = build_sign_timeline(gloss_with_emotion, cfg)
            st.success("✅ Done!")
            cache_mgr.save_stage_cache(video_id, "timeline", timeline)
        
        # Save files
        wav_stem = wav_path.stem
        save_gloss_json(gloss_with_emotion, wav_stem, cfg)
        save_timeline_json(timeline, wav_stem, cfg)
        
        # Store in session
        st.session_state.transcript_json = transcript_json
        st.session_state.gloss_with_emotion = gloss_with_emotion
        st.session_state.timeline = timeline
        st.session_state.cfg = cfg
        st.session_state.video_id = video_id
        st.session_state.stages_from_cache = stages_from_cache
        
        if stages_from_cache:
            st.success(f"⚡ {len(stages_from_cache)} stages loaded from cache!")
        st.balloons()
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


# ============================================================
# RESULTS
# ============================================================
if "transcript_json" in st.session_state:
    st.divider()
    st.header("📊 Results")
    
    # Status badge
    stages_from_cache = st.session_state.get("stages_from_cache", [])
    if stages_from_cache:
        cache_text = ", ".join(stages_from_cache)
        st.info(f"⚡ Stages from cache: {cache_text}")
    
    # Language
    language = st.session_state.transcript_json.get("language", "unknown").upper()
    st.markdown(f"🌐 **Language Detected:** {language}")
    
    # Video player
    st.subheader("🎬 Video Player")
    video_id = st.session_state.get("video_id", "")
    if video_id:
        st.markdown(f"""
        <iframe width="100%" height="500" 
                src="https://www.youtube.com/embed/{video_id}?start=0&cc_load_policy=1" 
                frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen></iframe>
        """, unsafe_allow_html=True)
        st.caption("💡 Enable CC button in player for captions")
    
    # Summary
    st.subheader("📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    timeline_data = st.session_state.timeline.get("timeline", [])
    with col1:
        st.metric("📝 Segments", len(timeline_data))
    with col2:
        st.metric("🤟 Sign Items", sum(len(s.get("items", [])) for s in timeline_data))
    with col3:
        text = st.session_state.transcript_json.get("text", "")
        st.metric("📄 Words", len(text.split()))
    with col4:
        duration = timeline_data[-1]["end"] if timeline_data else 0
        st.metric("⏱️ Duration", f"{duration:.1f}s")
    
    st.divider()
    
    # Transcript with ISL Images
    st.subheader("📚 Full Transcript with Real Sign Language Images")
    
    # Load ISL dataset
    isl_path = Path(__file__).parent / "data"
    isl_loader = get_isl_loader(isl_path) if isl_path.exists() else None
    
    for i, seg in enumerate(timeline_data, 1):
        with st.container():
            # Header with timestamp
            st.markdown(f"### Segment {i}: ⏱️ {seg['start']:.1f}s - {seg['end']:.1f}s")
            
            # English transcript
            st.markdown("**📝 English:**")
            st.markdown(f'> {seg["text"]}')
            
            # Emotion badge (if exists)
            if seg.get('emotion'):
                emo = seg['emotion']
                emojis = {'joy': '😊', 'sadness': '😢', 'anger': '😠', 'fear': '😨', 'neutral': '😐', 'surprise': '😮'}
                emoji = emojis.get(emo.get('label', 'neutral'), '😐')
                st.caption(f"{emoji} Emotion: {emo.get('label', 'neutral').upper()} ({emo.get('score', 0):.0%})")
            
            # Full sign language description
            gloss = seg.get('gloss', [])
            gloss_clean = [g for g in gloss if g != '|']
            
            if gloss_clean:
                # Full description
                st.markdown("**🤟 Sign Language (Full Description):**")
                gloss_text = " → ".join(gloss_clean)
                st.markdown(f'### {gloss_text}')
                
                # Initialize GIF generator
                gif_gen = get_dual_gif_generator()
                
                # ===== DISPLAY 1: LETTER-WISE GIFs (Individual letters) =====
                st.markdown("**📸 Letter-Wise Fingerspelling (Individual Letters):**")
                st.caption("🔤 Each letter individually - perfect for learning hand positions")
                
                try:
                    letter_cols = st.columns(len(gloss_clean) if len(gloss_clean) <= 4 else 4)
                    
                    for idx, token in enumerate(gloss_clean):
                        first_letter = token[0].upper() if token else '?'
                        col = letter_cols[idx % len(letter_cols)]
                        
                        with col:
                            # Generate or get cached letter GIF
                            letter_gif = gif_gen.create_letter_gif(first_letter)
                            
                            if letter_gif and Path(letter_gif).exists():
                                st.image(letter_gif, caption=f"🔤 {first_letter}", use_container_width=True)
                            else:
                                st.markdown(f"### {first_letter}\n*Loading...*")
                
                except Exception as e:
                    st.warning(f"⚠️ Could not generate letter GIFs: {str(e)}")
                
                st.divider()
                
                # ===== DISPLAY 2: TOKEN-WISE GIFs (Full words) =====
                st.markdown("**🎬 Token-Wise Fingerspelling (Full Words):**")
                st.caption("🎞️ Watch each word spelled out letter by letter at comfortable speed (300ms per letter)")
                
                try:
                    # Create GIF for each token
                    token_cols = st.columns(min(3, len(gloss_clean)))
                    
                    for idx, token in enumerate(gloss_clean):
                        col = token_cols[idx % len(token_cols)]
                        
                        with col:
                            # Generate or get cached token GIF
                            token_gif = gif_gen.create_token_gif(token)
                            
                            if token_gif and Path(token_gif).exists():
                                st.image(token_gif, caption=f"🎬 Spelling: {token}", use_container_width=True)
                            else:
                                st.markdown(f"### {token}\n*Generating...*")
                
                except Exception as e:
                    st.warning(f"⚠️ Could not generate token GIFs: {str(e)}")
                
                st.divider()
                
                # ===== DISPLAY 3: COMBINED GIF (All words fast) =====
                st.markdown("**⚡ Combined Fast Fingerspelling (All Words):**")
                st.caption("🚀 All words together at faster speed (250ms per letter) - challenge mode!")
                
                try:
                    gif_id = str(uuid.uuid4())[:8]
                    combined_gif = f"output/combined_{gif_id}.gif"
                    
                    create_fingerspelling_gif(gloss_clean, combined_gif, duration_per_letter=250)
                    
                    st.image(combined_gif, caption=f"⚡ Combined: {gloss_text}", use_container_width=True)
                    
                    Path(combined_gif).unlink(missing_ok=True)
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not create combined animation: {str(e)}")
                
                st.divider()
                
                # Display real ISL images (bonus)
                st.markdown("**📸 Individual Sign Language Images (Real ISL):**")
                
                if isl_loader:
                    # Display in rows of 4
                    for row_idx in range(0, len(gloss_clean), 4):
                        row_signs = gloss_clean[row_idx:min(row_idx + 4, len(gloss_clean))]
                        row_cols = st.columns(len(row_signs), gap="medium")
                        
                        for col, token in zip(row_cols, row_signs):
                            with col:
                                try:
                                    # Get first letter for image lookup
                                    first_letter = token[0].upper() if token else '?'
                                    
                                    # Load ISL image
                                    image = isl_loader.get_sign_image(first_letter, size=350)
                                    
                                    if image:
                                        col.image(image, caption=f"🤟 {token}", use_container_width=True)
                                    else:
                                        col.markdown(f"### {token}\n*ISL image not available*", help=f"No images for {first_letter}")
                                except Exception as e:
                                    col.markdown(f"### {token}\n*Error loading*")
                else:
                    st.warning("⚠️ ISL dataset not found at /data")
            else:
                st.info("⚠️ No sign language tokens for this segment")
            
            st.divider()
    
    st.divider()
    
    # Download
    st.subheader("💾 Download")
    col1, col2 = st.columns(2)
    
    with col1:
        transcript_text = st.session_state.transcript_json.get("text", "")
        st.download_button(
            "📋 Transcript (TXT)",
            data=transcript_text,
            file_name="transcript.txt",
            mime="text/plain"
        )
    
    with col2:
        timeline_json = json.dumps(st.session_state.timeline, indent=2)
        st.download_button(
            "🤟 Timeline (JSON)",
            data=timeline_json,
            file_name="sign_timeline.json",
            mime="application/json"
        )

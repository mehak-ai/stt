import streamlit as st
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tempfile
import os

# Ensure ffmpeg path is included (update if needed)
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-7.1.1-full_build\bin"

# Page config
st.set_page_config(page_title="🎤 Record or Upload & Transcribe", layout="centered")
st.title("🎙️ Whisper Speech-to-Text")

# ------------------- Sidebar Settings -------------------
with st.sidebar:
    st.header("🔧 Settings")
    language = st.selectbox("Spoken Language", ["auto", "en", "hi", "es", "fr", "de", "ja", "zh"])
    model_size = st.selectbox("Whisper Model Size", ["base", "small", "medium"], index=1)
    st.markdown("---")
    st.subheader("🎤 Recorder Settings")
    duration = st.slider("Recording Duration (seconds)", 1, 20, 5)
    sample_rate = st.selectbox("Sample Rate (Hz)", [16000, 22050, 44100], index=0)

# ------------------- Whisper Loader -------------------
@st.cache_resource(show_spinner=False)
def load_whisper_model(size):
    return whisper.load_model(size)

# ------------------- Audio Transcription -------------------
def transcribe_audio(path, lang):
    model = load_whisper_model(model_size)
    options = {}
    if lang != "auto":
        options["language"] = lang
    return model.transcribe(path, **options)

# ------------------- UI Tabs -------------------
tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎤 Record from Microphone"])

# ------------------- Upload UI -------------------
with tab1:
    audio_file = st.file_uploader("Upload Audio File (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])

    if audio_file is not None:
        st.audio(audio_file, format="audio/mp3")

        with st.spinner("🔍 Transcribing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            try:
                result = transcribe_audio(tmp_path, language)
                if result["text"].strip():
                    st.success("✅ Transcription complete!")
                    st.text_area("📄 Transcribed Text", result["text"], height=200)
                else:
                    st.warning("⚠️ No speech detected.")
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
            finally:
                os.remove(tmp_path)

# ------------------- Recorder UI -------------------
with tab2:
    if st.button("⏺️ Start Recording"):
        st.info("🎙️ Recording...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        st.success("✅ Recording complete!")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as tmpfile:
            write(tmpfile.name, sample_rate, audio)
            tmpfile.flush()
            os.fsync(tmpfile.fileno())
            tmpfile_path = tmpfile.name

        st.audio(tmpfile_path, format="audio/wav")
        with open(tmpfile_path, "rb") as f:
            st.download_button("⬇️ Download Audio", data=f.read(), file_name="recorded_audio.wav")

        with st.spinner("🔍 Transcribing..."):
            try:
                result = transcribe_audio(tmpfile_path, language)
                if result["text"].strip():
                    st.success("✅ Transcription complete!")
                    st.text_area("📄 Transcribed Text", result["text"], height=200)
                else:
                    st.warning("⚠️ No speech detected.")
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
            finally:
                os.remove(tmpfile_path)

import streamlit as st
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tempfile
import os
import subprocess
import time

# ----------------------------------
# Session state initialization
# ----------------------------------
if "record_run" not in st.session_state:
    st.session_state.record_run = 0

# Ensure ffmpeg path is included (update if needed)
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-7.1.1-full_build\bin"

# ------------------- FFmpeg Check -------------------
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        st.error("❌ FFmpeg not found. Install FFmpeg and add it to PATH.")
check_ffmpeg()

# ------------------- Extract Audio Function (Missing in your code) -------------------
def extract_audio_from_video(video_path):
    """
    Extract audio from a video file into a WAV (16kHz mono) and return the path.
    """
    import tempfile
    import subprocess
    import os

    # Create a proper temporary WAV file
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = tmp_audio.name
    tmp_audio.close()  # Close it so FFmpeg can write

    # FFmpeg command
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",               # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",      # 16 kHz
        "-ac", "1",          # mono
        "-y",                # overwrite
        audio_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr.decode()}")

    return audio_path


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
tab1, tab2, tab3 = st.tabs(["📁 Upload Audio File", "🎤 Record from Microphone", "🎬 Video / YouTube"])

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
    # Increment run counter on button click (forces reset)
    if st.button("⏺️ Start Recording"):
        st.session_state.record_run += 1

    # Placeholders (everything goes inside these)
    status_placeholder = st.empty()
    output_placeholder = st.container()

    # Only run recording after button click
    if st.session_state.record_run > 0:
        with output_placeholder:
            # Show recording status
            status_placeholder.info("🎙️ Recording...")

            # Start recording
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='int16'
            )
            sd.wait()

            # Remove recording message
            status_placeholder.empty()

            st.success("✅ Recording complete!")

            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as tmpfile:
                write(tmpfile.name, sample_rate, audio)
                tmpfile.flush()
                os.fsync(tmpfile.fileno())
                tmpfile_path = tmpfile.name

            # Play recorded audio
            st.audio(tmpfile_path, format="audio/wav")

            # Download button
            with open(tmpfile_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Audio",
                    data=f.read(),
                    file_name="recorded_audio.wav",
                    mime="audio/wav"
                )

            # Transcription
            with st.spinner("🔍 Transcribing..."):
                try:
                    result = transcribe_audio(tmpfile_path, language)

                    if result["text"].strip():
                        st.success("✅ Transcription complete!")
                        st.text_area(
                            "📄 Transcribed Text",
                            result["text"],
                            height=200
                        )
                    else:
                        st.warning("⚠️ No speech detected.")

                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")

                finally:
                    if os.path.exists(tmpfile_path):
                        os.remove(tmpfile_path)


# ------------------- Video / YouTube Transcription UI -------------------
with tab3:
    st.subheader("🎬 Upload Video or Paste YouTube Link")

    video_file = st.file_uploader("Upload Video (MP4 / MKV / MOV)", type=["mp4", "mkv", "mov"])
    video_url = st.text_input("Or Paste Video URL (YouTube)")

    # ------------------- Uploaded Video -------------------
    if video_file:
        st.video(video_file)

        # Save uploaded video safely on Windows
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        try:
            with st.spinner("🎧 Extracting audio..."):
                audio_path = extract_audio_from_video(video_path)

            with st.spinner("🔍 Transcribing..."):
                result = transcribe_audio(audio_path, language)
                if result["text"].strip():
                    st.success("✅ Video Transcription Complete!")
                    st.text_area("📄 Transcribed Text", result["text"], height=250)
                else:
                    st.warning("⚠️ No speech detected.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            # Cleanup temporary files
            if os.path.exists(video_path):
                os.remove(video_path)
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)

    # ------------------- YouTube Video -------------------
    if video_url:
        st.warning("📥 Downloading from YouTube requires 'yt-dlp' installed.")

        if st.button("⬇️ Download & Transcribe YouTube Video"):
            video_path = tempfile.mktemp(suffix=".mp4")  # safer on Windows

            # Download YouTube video
            st.info("📥 Downloading video...")
            try:
                subprocess.run(f'yt-dlp -f best -o "{video_path}" "{video_url}"', 
                               shell=True, check=True)
                st.success("✅ Download complete.")

                with st.spinner("🎧 Extracting audio..."):
                    audio_path = extract_audio_from_video(video_path)

                with st.spinner("🔍 Transcribing..."):
                    result = transcribe_audio(audio_path, language)
                    if result["text"].strip():
                        st.success("✅ Video Transcription Complete!")
                        st.text_area("📄 Transcribed Text", result["text"], height=250)
                    else:
                        st.warning("⚠️ No speech detected.")
            except subprocess.CalledProcessError as e:
                st.error(f"❌ YouTube download failed:\n{e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                # Cleanup temporary files
                if os.path.exists(video_path):
                    os.remove(video_path)
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)

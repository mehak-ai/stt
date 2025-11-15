import streamlit as st
import whisper
import tempfile
import os
import subprocess
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import soundfile as sf

# Ensure ffmpeg path is included if needed (mostly for local)
#os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-7.1.1-full_build\bin"

# ------------------- FFmpeg Check -------------------
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        st.error("❌ FFmpeg not found. Install FFmpeg and add it to PATH.")
check_ffmpeg()

# ------------------- Extract Audio Function -------------------
def extract_audio_from_video(video_path):
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio_path = tmp_audio.name
    tmp_audio.close()

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        audio_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr.decode()}")

    return audio_path

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

# ------------------- Page Config -------------------
st.set_page_config(page_title="🎤 Record or Upload & Transcribe", layout="centered")
st.title("🎙️ Whisper Speech-to-Text")

# ------------------- Sidebar -------------------
with st.sidebar:
    st.header("🔧 Settings")
    language = st.selectbox("Spoken Language", ["auto", "en", "hi", "es", "fr", "de", "ja", "zh"])
    model_size = st.selectbox("Whisper Model Size", ["base", "small", "medium"], index=1)
    st.markdown("---")
    st.subheader("🎤 Recorder Settings")
    duration = st.slider("Recording Duration (seconds)", 1, 20, 5)

# ------------------- UI Tabs -------------------
tab1, tab2, tab3 = st.tabs(["📁 Upload Audio File", "🎤 Record from Microphone", "🎬 Video / YouTube"])

# ------------------- Upload Audio -------------------
with tab1:
    audio_file = st.file_uploader("Upload Audio File (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])
    if audio_file:
        st.audio(audio_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        with st.spinner("🔍 Transcribing..."):
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

# ------------------- Browser-based Recorder -------------------
with tab2:
    st.subheader("🎤 Record from Microphone (Browser)")
    
    class AudioProcessor:
        def recv(self, frame: av.AudioFrame):
            # Convert audio frame to numpy array
            audio = frame.to_ndarray()
            # Convert stereo to mono if needed
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            self.audio = audio
            return frame

    webrtc_ctx = webrtc_streamer(
    key="microphone",
    mode=WebRtcMode.RECVONLY,
    media_stream_constraints={"audio": True, "video": False},  # <- use this instead
    audio_receiver_size=1024,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


    if webrtc_ctx.state.playing:
        if st.button("⏺️ Stop & Transcribe"):
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=duration)
            if audio_frames:
                combined_audio = np.concatenate([f.to_ndarray().mean(axis=0) if f.to_ndarray().ndim > 1 else f.to_ndarray() for f in audio_frames])
                # Save to temp WAV
                tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(tmp_wav.name, combined_audio, 16000)
                tmp_wav_path = tmp_wav.name
                tmp_wav.close()
                
                st.audio(tmp_wav_path)
                with open(tmp_wav_path, "rb") as f:
                    st.download_button("⬇️ Download Audio", data=f.read(), file_name="recorded_audio.wav")
                
                # Transcribe
                with st.spinner("🔍 Transcribing..."):
                    try:
                        result = transcribe_audio(tmp_wav_path, language)
                        if result["text"].strip():
                            st.success("✅ Transcription complete!")
                            st.text_area("📄 Transcribed Text", result["text"], height=200)
                        else:
                            st.warning("⚠️ No speech detected.")
                    except Exception as e:
                        st.error(f"❌ Transcription failed: {e}")
                    finally:
                        os.remove(tmp_wav_path)
            else:
                st.warning("⚠️ No audio frames captured.")

# ------------------- Video / YouTube -------------------
with tab3:
    st.subheader("🎬 Upload Video or Paste YouTube Link")
    video_file = st.file_uploader("Upload Video (MP4 / MKV / MOV)", type=["mp4", "mkv", "mov"])
    video_url = st.text_input("Or Paste Video URL (YouTube)")

    if video_file:
        st.video(video_file)
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        try:
            audio_path = extract_audio_from_video(video_path)
            result = transcribe_audio(audio_path, language)
            if result["text"].strip():
                st.success("✅ Video Transcription Complete!")
                st.text_area("📄 Transcribed Text", result["text"], height=250)
            else:
                st.warning("⚠️ No speech detected.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)

    if video_url:
        st.warning("📥 Downloading from YouTube requires 'yt-dlp' installed.")
        if st.button("⬇️ Download & Transcribe YouTube Video"):
            video_path = tempfile.mktemp(suffix=".mp4")
            try:
                subprocess.run(f'yt-dlp -f best -o "{video_path}" "{video_url}"', shell=True, check=True)
                audio_path = extract_audio_from_video(video_path)
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
                if os.path.exists(video_path):
                    os.remove(video_path)
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)

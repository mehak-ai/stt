import streamlit as st
import whisper
import tempfile
import os
import subprocess
import shutil
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import soundfile as sf

# ------------------- FFmpeg Check -------------------
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        st.error("❌ FFmpeg not found. Install FFmpeg and add it to PATH.")
check_ffmpeg()

# ------------------- Extract Audio From Video -------------------
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
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr.decode(errors='ignore')}")

    return audio_path

# ------------------- YouTube download (uses yt-dlp CLI) -------------------
def download_youtube_video(url, cookie_path):
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_path = tmp_video.name
    tmp_video.close()

    if not shutil.which("yt-dlp"):
        raise RuntimeError("❌ 'yt-dlp' not found. Install yt-dlp and ensure it's in PATH.")

    cmd = [
        "yt-dlp",
        "--cookies", cookie_path,
        "--extractor-args", "youtube:player_client=default",
        "--geo-bypass",
        "-f", "bestaudio/best",
        "-o", video_path,
        url
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error:\n{result.stderr}")

    if not os.path.exists(video_path) or os.path.getsize(video_path) < 50000:
        raise RuntimeError(
            "❌ YouTube download failed.\n"
            "Invalid/empty file.\n"
            "➡️ Your cookies.txt is expired OR YouTube blocked the server."
        )

    return video_path

# ------------------- Whisper Loader -------------------
@st.cache_resource(show_spinner=False)
def load_whisper_model(size):
    return whisper.load_model(size)

# ------------------- Audio Transcription -------------------
def transcribe_audio(path, lang, model_size):
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

# ===================== TAB 1: UPLOAD AUDIO =====================
with tab1:
    audio_file = st.file_uploader("Upload Audio File (MP3/WAV/M4A)", type=["mp3", "wav", "m4a"])
    
    if audio_file:
        st.audio(audio_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Transcribing..."):
            try:
                result = transcribe_audio(tmp_path, language, model_size)
                st.success("✅ Transcription complete!")
                st.text_area("📄 Transcribed Text", result.get("text", ""), height=200)
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
            finally:
                os.remove(tmp_path)

# ===================== TAB 2: BROWSER MICROPHONE RECORDING =====================
with tab2:
    st.subheader("🎤 Record from Microphone (Browser)")

    webrtc_ctx = webrtc_streamer(
        key="mic-recorder",
        mode=WebRtcMode.RECVONLY,
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    if webrtc_ctx.state.playing:
        if st.button("⏺️ Stop & Transcribe"):
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=duration)

            if audio_frames:
                try:
                    arrays = []
                    for frame in audio_frames:
                        arr = frame.to_ndarray()
                        if arr.ndim > 1:
                            arr = arr.mean(axis=0)
                        arrays.append(arr)

                    combined_audio = np.concatenate(arrays)

                    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    wav_path = tmp_wav.name
                    tmp_wav.close()

                    sf.write(wav_path, combined_audio, 16000)

                    st.audio(wav_path)

                    with st.spinner("🔍 Transcribing..."):
                        result = transcribe_audio(wav_path, language, model_size)
                        st.success("✅ Transcription complete!")
                        st.text_area("📄 Transcribed Text", result.get("text", ""), height=200)

                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")

                finally:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
            else:
                st.warning("⚠️ No audio frames captured.")

# ===================== TAB 3: VIDEO + YOUTUBE =====================
with tab3:
    st.subheader("🎬 Upload Video or Paste YouTube Link")

    # ---------------- NEW: Cookies Upload ----------------
    uploaded_cookies = st.file_uploader("Upload your cookies.txt (Required for YouTube)", type=["txt"])

    cookie_path = None
    if uploaded_cookies:
        cookie_path = "cookies.txt"
        with open(cookie_path, "wb") as f:
            f.write(uploaded_cookies.read())
        st.success("✅ Cookies uploaded successfully!")

    video_file = st.file_uploader("Upload Video (MP4/MKV/MOV)", type=["mp4", "mkv", "mov"])
    video_url = st.text_input("Or Paste Video URL (YouTube)")

    # ---------- Uploaded Video ----------
    if video_file:
        st.video(video_file)

        video_path = tempfile.mktemp(suffix=".mp4")
        audio_path = None

        with open(video_path, "wb") as f:
            f.write(video_file.read())

        try:
            with st.spinner("🎧 Extracting audio..."):
                audio_path = extract_audio_from_video(video_path)

            with st.spinner("🔍 Transcribing..."):
                result = transcribe_audio(audio_path, language, model_size)

            st.success("✅ Video Transcription Complete!")
            st.text_area("📄 Transcribed Text", result.get("text", ""), height=250)

        except Exception as e:
            st.error(f"❌ Error: {e}")

        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

    # ---------- YouTube ----------
    # if video_url:
    #     if not cookie_path:
    #         st.error("❌ You must upload cookies.txt to download YouTube videos.")
    #     else:
    #         if st.button("⬇️ Download & Transcribe YouTube Video"):
    #             video_path = None
    #             audio_path = None
    #
    #             try:
    #                 with st.spinner("📥 Downloading video..."):
    #                     video_path = download_youtube_video(video_url, cookie_path)
    #
    #                 with st.spinner("🎧 Extracting audio..."):
    #                     audio_path = extract_audio_from_video(video_path)
    #
    #                 with st.spinner("🔍 Transcribing..."):
    #                     result = transcribe_audio(audio_path, language, model_size)
    #
    #                 st.success("✅ Video Transcription Complete!")
    #                 st.text_area("📄 Transcribed Text", result.get("text", ""), height=250)
    #
    #             except Exception as e:
    #                 st.error(f"❌ Error: {e}")
    #
    #             finally:
    #                 if video_path and os.path.exists(video_path):
    #                     os.remove(video_path)
    #                 if audio_path and os.path.exists(audio_path):
    #                     os.remove(audio_path)



# 🎙️ Whisper Speech-to-Text App


A web application built with **Streamlit** and **OpenAI Whisper** for transcribing speech from audio, video, and YouTube videos. It supports multiple languages and provides easy downloading of transcriptions and recorded audio.



## Features

* **Upload Audio Files**: Supports MP3, WAV, and M4A files.
* **Record from Browser Microphone** (local only): Uses WebRTC to record audio directly in your browser.
* **Upload Video Files**: Supports MP4, MKV, MOV formats and extracts audio for transcription.
* **YouTube Video Transcription**: Paste a YouTube link and the app will download and transcribe it.
* **Downloadable Files**: Download your recorded audio or extracted audio from videos.
* **Real-time Feedback**: Shows transcription progress using spinners and success messages.




## Installation (Local)

```bash
# Clone the repository
git clone <your-repo-link>
cd <repo-folder>

# Create a virtual environment
conda create -n stt python=3.10
conda activate stt

# Install required packages
pip install -r requirements.txt

# Run the app
streamlit run recordstreamlit.py
```



## Requirements

* Python 3.10+
* Streamlit
* OpenAI Whisper (`pip install openai-whisper`)
* SoundFile (`pip install soundfile`)
* NumPy, SciPy
* FFmpeg (must be installed and added to PATH for local use)
* yt-dlp (for YouTube video downloads)

> ⚠️ **Note for Streamlit Cloud Deployment**: Browser microphone recording using `streamlit-webrtc` is **not supported** on Streamlit Cloud. Use file uploads for recording instead.



## Usage

1. Open the app.
2. Select the spoken language from the sidebar.
3. Choose one of the tabs:

   * **Upload Audio File**: Upload your audio and transcribe.
   * **Record from Microphone**: Record audio locally (not supported on cloud).
   * **Video / YouTube**: Upload a video or paste a YouTube URL to transcribe.
4. Download your audio if needed.
5. View or copy the transcribed text.



## License

This project is **open source** and free to use for personal and educational purposes.



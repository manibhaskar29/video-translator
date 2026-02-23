# 🎥 Video Translator & Dubber

Automatically download YouTube videos and dub them into another language. The project ships with **two pipelines**:

| Script | Description | Translation Engine | TTS Engine |
|---|---|---|---|
| `translator.py` | **Multi-language** — auto-detects source, translates to English / Telugu / Hindi | Google Translate | Edge TTS (neural voice) |
| `pro_dubber.py` | **Pro** — Hindi → Telugu with English words preserved | Gemini AI (+ Google Translate fallback) | Edge TTS (neural voice) |

Both scripts use **faster-whisper `large-v3`** on GPU for accurate transcription and **timestamp-aligned audio stitching** for natural output.

---

## ⚡ Prerequisites

| Tool | Why |
|---|---|
| **Python 3.10+** | Runtime |
| **ffmpeg** | Audio extraction & video merging (must be on `PATH`) |
| **NVIDIA GPU + CUDA** | Whisper `large-v3` runs on GPU with `float16` |

### Install ffmpeg (Windows)

```bash
winget install --id Gyan.FFmpeg -e --source winget
```

Or download from <https://ffmpeg.org/download.html> and add the `bin` folder to your system `PATH`.

---

## 🚀 Setup

```bash
# 1. Clone the repo
git clone https://github.com/manibhaskar29/video-translator.git
cd video-translator

# 2. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Set up your Gemini API key *(pro_dubber only)*

1. Get a free API key from <https://aistudio.google.com/apikey>
2. Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## 🏃 How to Run

### Multi-Language Translator (`translator.py`)

```bash
python translator.py
```

1. Paste a YouTube URL when prompted.
2. Choose a target language (English / Telugu / Hindi).
3. The pipeline auto-detects the source language, transcribes with **Whisper large-v3** on GPU, translates, and generates timestamp-aligned audio.
4. Output is saved as **`output_en.mp4`** / **`output_te.mp4`** / **`output_hi.mp4`**.

### Pro Dubber (`pro_dubber.py`)

```bash
python pro_dubber.py
```

1. Paste a YouTube URL when prompted.
2. The pipeline will:
   - Download the video via `yt-dlp`
   - Extract audio with `ffmpeg`
   - Transcribe using **Whisper large-v3** on GPU
   - Translate each segment with **Gemini AI** (falls back to Google Translate on rate limits)
   - Generate natural Telugu speech with **Edge TTS**
   - Stitch audio aligned to the original timestamps
   - Merge the new audio back into the video
3. The final dubbed video is saved as **`output_telugu.mp4`**.

#### Resume mode

If the script was interrupted after transcription, just run it again — it detects `transcription.json` and offers to resume from the translation step:

```bash
python pro_dubber.py --resume
```

---

## 📁 Project Structure

```
Video Translator/
├── pro_dubber.py          # Pro pipeline (GPU + Gemini + Edge TTS)
├── translator.py          # Basic pipeline (Whisper base + Google Translate + gTTS)
├── requirements.txt       # Python dependencies
├── .env                   # Gemini API key (not committed)
├── .gitignore
└── README.md
```

### Generated files (gitignored)

| File | Description |
|---|---|
| `video.mp4` | Downloaded source video |
| `audio.wav` | Extracted audio |
| `transcription.json` | Whisper transcription with timestamps |
| `original_text.txt` | Original transcribed text |
| `translated.json` | Translated segments with timestamps |
| `translated.txt` | Translated text |
| `output.mp4` / `output_telugu.mp4` | Final dubbed video |

---

## 🛠 Tech Stack

- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — YouTube downloading
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — GPU-accelerated speech-to-text
- **[Gemini AI](https://ai.google.dev/)** — Intelligent translation (preserves English terms)
- **[Edge TTS](https://github.com/rany2/edge-tts)** — Microsoft neural text-to-speech
- **[pydub](https://github.com/jiaaro/pydub)** — Audio manipulation & timestamp stitching
- **[ffmpeg](https://ffmpeg.org/)** — Audio/video processing

---

## 📝 License

This project is for educational and personal use.

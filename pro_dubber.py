"""
🎥 Pro Video Dubber — Production Pipeline
Hindi → Telugu (English words preserved)

Architecture:
  YouTube → yt-dlp → faster-whisper large-v3 (GPU)
  → Gemini AI translation → edge-tts Telugu voice
  → timestamp-aligned audio stitching → ffmpeg merge
"""

import yt_dlp
import subprocess
import os
import asyncio
import json
import time
import gc
import sys
from pathlib import Path
from faster_whisper import WhisperModel
from pydub import AudioSegment
from google import genai
import edge_tts
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# ========== CONFIG ==========
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
WHISPER_MODEL = "large-v3"        # Best accuracy for Hindi
WHISPER_DEVICE = "cuda"           # GPU acceleration
WHISPER_COMPUTE = "float16"       # Optimized for 6GB VRAM
TELUGU_VOICE = "te-IN-ShrutiNeural"  # Natural Telugu female voice
# Other options: "te-IN-MohanNeural" (male)

TEMP_DIR = "temp_segments"
OUTPUT_FILE = "output_telugu.mp4"

# ========== SETUP ==========
if not GEMINI_API_KEY or GEMINI_API_KEY == "paste_your_key_here":
    print("❌ ERROR: Please add your Gemini API key to .env file!")
    print("   Get free key from: https://aistudio.google.com/apikey")
    print("   Then edit .env and paste your key")
    exit(1)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash"


# ========== STEP 1: Download Video ==========
def download_video(url):
    print("\n📥 Step 1/6: Downloading video...")

    if os.path.exists("video.mp4"):
        os.remove("video.mp4")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'video.mp4',
        'merge_output_format': 'mp4',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print("✅ Video downloaded!")


# ========== STEP 2: Extract Audio ==========
def extract_audio():
    print("\n🔊 Step 2/6: Extracting audio...")

    command = [
        "ffmpeg", "-y",
        "-i", "video.mp4",
        "-ar", "16000",
        "-ac", "1",
        "audio.wav"
    ]
    subprocess.run(command, check=True, capture_output=True)

    print("✅ Audio extracted!")


# ========== STEP 3: Transcribe with GPU ==========
def transcribe_gpu():
    print(f"\n🗣️ Step 3/6: Transcribing with Whisper {WHISPER_MODEL} on GPU...")
    print("   (First run downloads the model ~3GB, subsequent runs are instant)")

    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE
    )

    segments, info = model.transcribe(
        "audio.wav",
        language="hi",
        vad_filter=True,            # Filter out silence
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    segment_list = []
    for segment in segments:
        seg_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        }
        segment_list.append(seg_data)
        print(f"   [{seg_data['start']:.1f}s - {seg_data['end']:.1f}s] {seg_data['text']}")

    # Save transcription for debugging
    with open("transcription.json", "w", encoding="utf-8") as f:
        json.dump(segment_list, f, ensure_ascii=False, indent=2)

    # Also save plain text
    full_text = "\n".join([s["text"] for s in segment_list])
    with open("original_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\n✅ Transcribed {len(segment_list)} segments!")
    print(f"📝 Saved to transcription.json and original_text.txt")

    # Free GPU memory
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("   GPU memory freed.")
    except Exception as e:
        print(f"   ⚠️ GPU cleanup warning (non-fatal): {e}")

    return segment_list


# ========== STEP 4: Intelligent Translation ==========
def translate_with_google(text):
    """Fallback: Use Google Translate when Gemini is rate-limited."""
    try:
        translated = GoogleTranslator(source='hi', target='te').translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"   ⚠️ Google Translate also failed: {e}")
        return text


def translate_segment(text, retries=5):
    """Translate one segment: Hindi → Telugu, English stays English.
    Uses Gemini AI first (best quality), falls back to Google Translate."""

    prompt = f"""Translate the following text from Hindi to Telugu.

CRITICAL RULES:
1. Translate ONLY the Hindi words into Telugu script.
2. Keep ALL English words EXACTLY as they are — do NOT translate or transliterate them.
3. Keep technical terms (Python, AI, ML, JavaScript, API, etc.) in English.
4. Keep proper nouns and brand names in English.
5. Return ONLY the translated text, no explanations or quotes.

Text: {text}"""

    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
            if response.text:
                result = response.text.strip()
                # Remove any quotes Gemini might add
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
                if result.startswith("'") and result.endswith("'"):
                    result = result[1:-1]
                return result
            else:
                print(f"   ⚠️ Empty response, retrying...")
                time.sleep(5)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < 2:  # Try Gemini twice
                    wait_time = (attempt + 1) * 15
                    print(f"   ⏳ Rate limited. Waiting {wait_time}s... (attempt {attempt+1}/{retries})")
                    time.sleep(wait_time)
                else:
                    # Fall back to Google Translate
                    print(f"   🔄 Switching to Google Translate fallback...")
                    return translate_with_google(text)
            else:
                wait_time = (attempt + 1) * 5
                print(f"   ⚠️ Gemini error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    # All retries exhausted — use Google Translate
    print(f"   🔄 Gemini exhausted. Using Google Translate fallback.")
    return translate_with_google(text)


def translate_all_segments(segments):
    print(f"\n🌐 Step 4/6: Translating {len(segments)} segments with Gemini AI...")
    print("   (Hindi → Telugu, English words preserved)")

    translated_segments = []

    for i, seg in enumerate(segments):
        original = seg["text"]
        translated = translate_segment(original)
        # Ensure we always have a string
        if not translated:
            translated = original

        translated_seg = {
            "start": seg["start"],
            "end": seg["end"],
            "original": original,
            "translated": translated
        }
        translated_segments.append(translated_seg)

        print(f"   [{i + 1}/{len(segments)}] {original[:40]}...")
        print(f"           → {translated[:40]}...")

        # Rate limiting: Gemini free tier = ~15 req/min
        # Use 4s delay between requests to stay safe
        if (i + 1) % 10 == 0:
            print(f"   ⏳ Pausing 30s for rate limit ({i + 1}/{len(segments)} done)...")
            time.sleep(30)
        else:
            time.sleep(4)

    # Save translations
    with open("translated.json", "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)

    translated_text = "\n".join([s["translated"] for s in translated_segments])
    with open("translated.txt", "w", encoding="utf-8") as f:
        f.write(translated_text)

    print(f"\n✅ All segments translated!")
    print(f"📝 Saved to translated.json and translated.txt")

    return translated_segments


# ========== STEP 5: Generate Telugu Audio ==========
async def generate_segment_audio(text, filename, voice=TELUGU_VOICE):
    """Generate audio for one segment using edge-tts."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)


async def generate_all_audio(translated_segments):
    print(f"\n🔈 Step 5/6: Generating Telugu audio for {len(translated_segments)} segments...")

    os.makedirs(TEMP_DIR, exist_ok=True)

    for i, seg in enumerate(translated_segments):
        filename = os.path.join(TEMP_DIR, f"seg_{i:04d}.mp3")
        text = seg["translated"]

        try:
            await generate_segment_audio(text, filename)
            print(f"   [{i + 1}/{len(translated_segments)}] ✓ Generated audio")
        except Exception as e:
            print(f"   [{i + 1}/{len(translated_segments)}] ❌ Failed: {e}")
            # Create a short silence as fallback
            silence = AudioSegment.silent(duration=1000)
            silence.export(filename, format="mp3")

    print("✅ All audio segments generated!")


def stitch_audio_with_timestamps(translated_segments):
    """Stitch all segment audios into one timeline-aligned track."""

    print("\n   🧵 Stitching audio with timestamp alignment...")

    # Get video duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", "video.mp4"],
        capture_output=True, text=True
    )
    video_duration_ms = int(float(result.stdout.strip()) * 1000)

    # Start with silence for the full video duration
    final_audio = AudioSegment.silent(duration=video_duration_ms)

    for i, seg in enumerate(translated_segments):
        filename = os.path.join(TEMP_DIR, f"seg_{i:04d}.mp3")

        if not os.path.exists(filename):
            continue

        try:
            segment_audio = AudioSegment.from_file(filename)
        except Exception:
            continue

        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        available_duration = end_ms - start_ms

        # If TTS audio is longer than the original segment time,
        # speed it up slightly to fit
        if len(segment_audio) > available_duration and available_duration > 0:
            speed_factor = len(segment_audio) / available_duration
            if speed_factor <= 2.0:  # Only speed up if reasonable
                segment_audio = segment_audio.speedup(
                    playback_speed=speed_factor,
                    chunk_size=50,
                    crossfade=25
                )

        # Overlay at the correct timestamp
        final_audio = final_audio.overlay(segment_audio, position=start_ms)

    final_audio.export("final_telugu_audio.wav", format="wav")
    print("✅ Timeline-aligned audio created!")

    return video_duration_ms


# ========== STEP 6: Merge with Video ==========
def merge_video():
    print("\n🎬 Step 6/6: Merging audio with video...")

    command = [
        "ffmpeg", "-y",
        "-i", "video.mp4",
        "-i", "final_telugu_audio.wav",
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        OUTPUT_FILE
    ]
    subprocess.run(command, check=True, capture_output=True)

    print(f"✅ Final video created: {OUTPUT_FILE}")


# ========== CLEANUP ==========
def cleanup():
    print("\n🧹 Cleaning up temporary files...")

    # Remove temp segment files
    if os.path.exists(TEMP_DIR):
        for f in os.listdir(TEMP_DIR):
            os.remove(os.path.join(TEMP_DIR, f))
        os.rmdir(TEMP_DIR)

    # Remove intermediate files
    for f in ["audio.wav", "final_telugu_audio.wav"]:
        if os.path.exists(f):
            os.remove(f)

    print("✅ Cleanup done!")


# ========== MAIN ==========
async def main():
    print("=" * 55)
    print("   🎥 PRO VIDEO DUBBER 🎥")
    print("   Hindi → Telugu (English words preserved)")
    print("   GPU Accelerated | Gemini AI | Natural Voice")
    print("=" * 55)

    # Check for resume mode
    resume_mode = "--resume" in sys.argv or os.path.exists("transcription.json")

    if resume_mode and os.path.exists("transcription.json") and os.path.exists("video.mp4"):
        print("\n🔄 Resume mode: Found existing transcription.json")
        print("   Skipping download, extraction, and transcription.")
        use_resume = input("   Resume from translation step? (y/n): ").strip().lower()

        if use_resume == "y" or use_resume == "yes":
            with open("transcription.json", "r", encoding="utf-8") as f:
                segments = json.load(f)
            print(f"   Loaded {len(segments)} segments from transcription.json")

            start_time = time.time()

            try:
                # Step 4: Translate with Gemini
                print("\n" + "-" * 55)
                translated_segments = translate_all_segments(segments)

                # Step 5: Generate Telugu audio
                await generate_all_audio(translated_segments)

                # Step 6: Stitch audio with timestamps
                video_duration_ms = stitch_audio_with_timestamps(translated_segments)

                # Step 7: Merge with video
                merge_video()

                # Cleanup
                cleanup()

                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)

                print()
                print("=" * 55)
                print(f"🎉 DONE! Processing took {minutes}m {seconds}s")
                print(f"📁 Output: {OUTPUT_FILE}")
                print(f"📊 Video duration: {video_duration_ms / 1000:.0f}s")
                print(f"📝 Segments processed: {len(translated_segments)}")
                print("=" * 55)

            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
            return

    # Full pipeline mode
    url = input("\nEnter YouTube video URL: ").strip()
    if not url:
        print("❌ No URL provided.")
        return

    print(f"\n🎯 Target: Hindi → Telugu (English preserved)")
    print(f"🖥️ Using: Whisper {WHISPER_MODEL} on {WHISPER_DEVICE.upper()}")
    print(f"🗣️ Voice: {TELUGU_VOICE}")
    print("-" * 55)

    start_time = time.time()

    try:
        # Step 1: Download
        download_video(url)
        print("   ✓ Step 1 complete")

        # Step 2: Extract audio
        extract_audio()
        print("   ✓ Step 2 complete")

        # Step 3: Transcribe with GPU
        segments = transcribe_gpu()
        print("   ✓ Step 3 complete")

        if not segments:
            print("❌ No speech detected in video!")
            return

        # Step 4: Translate with Gemini
        translated_segments = translate_all_segments(segments)
        print("   ✓ Step 4 complete")

        # Step 5: Generate Telugu audio
        await generate_all_audio(translated_segments)
        print("   ✓ Step 5 complete")

        # Step 6: Stitch audio with timestamps
        video_duration_ms = stitch_audio_with_timestamps(translated_segments)
        print("   ✓ Step 6 complete")

        # Step 7: Merge with video
        merge_video()

        # Cleanup
        cleanup()

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print()
        print("=" * 55)
        print(f"🎉 DONE! Processing took {minutes}m {seconds}s")
        print(f"📁 Output: {OUTPUT_FILE}")
        print(f"📊 Video duration: {video_duration_ms / 1000:.0f}s")
        print(f"📝 Segments processed: {len(translated_segments)}")
        print("=" * 55)

        print("\n📂 Files created:")
        print(f"   {OUTPUT_FILE}          — Final dubbed video")
        print(f"   transcription.json     — Whisper transcription + timestamps")
        print(f"   original_text.txt      — Original Hindi/English text")
        print(f"   translated.json        — Translated segments + timestamps")
        print(f"   translated.txt         — Translated Telugu text")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

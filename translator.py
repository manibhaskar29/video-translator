"""
🎥 Video Translator — Multi-Language Pipeline
Auto-detect source → Translate to English / Telugu / Hindi
Uses: faster-whisper large-v3 (GPU) + Google Translate + Edge TTS
"""

import yt_dlp
import subprocess
import os
import asyncio
import json
import gc
from pydub import AudioSegment
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import edge_tts

# ========== CONFIG ==========
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

# Edge TTS voice map — natural neural voices
VOICE_MAP = {
    "en": "en-US-AriaNeural",
    "te": "te-IN-ShrutiNeural",
    "hi": "hi-IN-SwaraNeural",
}

TEMP_DIR = "temp_segments"

# Whisper language codes → Google Translate codes
WHISPER_TO_GOOGLE = {
    'en': 'en', 'hi': 'hi', 'te': 'te', 'ta': 'ta',
    'mr': 'mr', 'bn': 'bn', 'gu': 'gu', 'kn': 'kn',
    'ml': 'ml', 'pa': 'pa', 'ur': 'ur',
}


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


# ========== STEP 3: Transcribe with GPU (faster-whisper large-v3) ==========
def transcribe_audio():
    print(f"\n🗣️ Step 3/6: Transcribing with Whisper {WHISPER_MODEL} on GPU...")
    print("   (First run downloads ~3GB model, subsequent runs are instant)")

    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )

    # Auto-detect language from first 30s
    segments_iter, info = model.transcribe(
        "audio.wav",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    detected_lang = info.language
    confidence = info.language_probability
    print(f"   🔍 Detected language: {detected_lang} ({confidence:.0%})")

    segment_list = []
    for seg in segments_iter:
        seg_data = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        }
        segment_list.append(seg_data)
        print(f"   [{seg_data['start']:.1f}s - {seg_data['end']:.1f}s] {seg_data['text'][:80]}")

    # Save transcription
    with open("transcription.json", "w", encoding="utf-8") as f:
        json.dump(segment_list, f, ensure_ascii=False, indent=2)

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
    except Exception:
        pass

    return segment_list, detected_lang


# ========== STEP 4: Translate Segments ==========
def translate_segments(segments, source_lang, target_lang):
    print(f"\n🌐 Step 4/6: Translating {len(segments)} segments ({source_lang} → {target_lang})...")

    if source_lang == target_lang:
        print("⚠️ Source and target are the same — skipping translation.")
        for seg in segments:
            seg["translated"] = seg["text"]
        return segments

    google_source = WHISPER_TO_GOOGLE.get(source_lang, "auto")

    translated_segments = []
    for i, seg in enumerate(segments):
        original = seg["text"]

        try:
            translated = GoogleTranslator(
                source=google_source, target=target_lang
            ).translate(original)
            if not translated:
                translated = original
        except Exception as e:
            print(f"   ⚠️ Segment {i+1} failed ({e}), trying auto-detect...")
            try:
                translated = GoogleTranslator(
                    source="auto", target=target_lang
                ).translate(original)
                if not translated:
                    translated = original
            except Exception:
                translated = original

        translated_seg = {
            "start": seg["start"],
            "end": seg["end"],
            "original": original,
            "translated": translated,
        }
        translated_segments.append(translated_seg)
        print(f"   [{i+1}/{len(segments)}] {original[:40]}...")
        print(f"           → {translated[:40]}...")

    # Save translations
    with open("translated.json", "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)

    with open("translated.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(s["translated"] for s in translated_segments))

    print(f"\n✅ All {len(translated_segments)} segments translated!")
    print(f"📝 Saved to translated.json and translated.txt")

    return translated_segments


# ========== STEP 5: Generate Audio (Edge TTS + Timestamp Stitching) ==========
async def generate_segment_audio(text, filename, voice):
    """Generate audio for one segment."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)


async def generate_all_audio(translated_segments, target_lang):
    print(f"\n🔈 Step 5/6: Generating audio for {len(translated_segments)} segments...")

    voice = VOICE_MAP.get(target_lang, "en-US-AriaNeural")
    print(f"   🗣️ Voice: {voice}")

    os.makedirs(TEMP_DIR, exist_ok=True)

    for i, seg in enumerate(translated_segments):
        filename = os.path.join(TEMP_DIR, f"seg_{i:04d}.mp3")
        text = seg["translated"]

        try:
            await generate_segment_audio(text, filename, voice)
            print(f"   [{i+1}/{len(translated_segments)}] ✓ Generated")
        except Exception as e:
            print(f"   [{i+1}/{len(translated_segments)}] ❌ Failed: {e}")
            silence = AudioSegment.silent(duration=1000)
            silence.export(filename, format="mp3")

    print("✅ All audio segments generated!")


def stitch_audio_with_timestamps(translated_segments):
    """Stitch segment audios into one timeline-aligned track."""
    print("\n   🧵 Stitching audio with timestamp alignment...")

    # Get video duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", "video.mp4"],
        capture_output=True, text=True,
    )
    video_duration_ms = int(float(result.stdout.strip()) * 1000)

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

        # Speed up if TTS audio is longer than original segment
        if len(segment_audio) > available_duration and available_duration > 0:
            speed_factor = len(segment_audio) / available_duration
            if speed_factor <= 2.0:
                segment_audio = segment_audio.speedup(
                    playback_speed=speed_factor,
                    chunk_size=50,
                    crossfade=25,
                )

        final_audio = final_audio.overlay(segment_audio, position=start_ms)

    final_audio.export("final_audio.wav", format="wav")
    print("✅ Timeline-aligned audio created!")

    return video_duration_ms


# ========== STEP 6: Merge with Video ==========
def merge_video(output_name):
    print("\n🎬 Step 6/6: Merging audio with video...")

    command = [
        "ffmpeg", "-y",
        "-i", "video.mp4",
        "-i", "final_audio.wav",
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        output_name,
    ]
    subprocess.run(command, check=True, capture_output=True)

    print(f"✅ Final video created: {output_name}")


# ========== CLEANUP ==========
def cleanup():
    print("\n🧹 Cleaning up temporary files...")

    if os.path.exists(TEMP_DIR):
        for f in os.listdir(TEMP_DIR):
            os.remove(os.path.join(TEMP_DIR, f))
        os.rmdir(TEMP_DIR)

    for f in ["audio.wav", "final_audio.wav"]:
        if os.path.exists(f):
            os.remove(f)

    print("✅ Cleanup done!")


# ========== MAIN ==========
async def main():
    print("=" * 55)
    print("   🎥 VIDEO TRANSLATOR 🎥")
    print("   Auto-detect → English / Telugu / Hindi")
    print("   GPU Accelerated | Edge TTS | Timestamp-Aligned")
    print("=" * 55)

    url = input("\nEnter YouTube video URL: ").strip()
    if not url:
        print("❌ No URL provided. Exiting.")
        return

    print("\nChoose target language:")
    print("  1. English")
    print("  2. Telugu")
    print("  3. Hindi")

    choice = input("\nEnter choice (1, 2, or 3): ").strip()

    lang_map = {
        "1": ("en", "English"),
        "2": ("te", "Telugu"),
        "3": ("hi", "Hindi"),
    }

    if choice not in lang_map:
        print("❌ Invalid choice. Exiting.")
        return

    target, lang_name = lang_map[choice]
    output_name = f"output_{target}.mp4"

    print(f"\n🚀 Starting: Auto-detect → {lang_name}")
    print(f"🖥️ Using: Whisper {WHISPER_MODEL} on {WHISPER_DEVICE.upper()}")
    print(f"🗣️ Voice: {VOICE_MAP.get(target, 'default')}")
    print("-" * 55)

    import time
    start_time = time.time()

    try:
        # Step 1
        download_video(url)

        # Step 2
        extract_audio()

        # Step 3
        segments, detected_lang = transcribe_audio()

        if not segments:
            print("❌ No speech detected in video!")
            return

        print(f"\n🔍 Video language: {detected_lang}")

        # Step 4
        translated_segments = translate_segments(segments, detected_lang, target)

        # Step 5
        await generate_all_audio(translated_segments, target)
        video_duration_ms = stitch_audio_with_timestamps(translated_segments)

        # Step 6
        merge_video(output_name)

        # Cleanup
        cleanup()

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print()
        print("=" * 55)
        print(f"🎉 DONE! Your {lang_name} video is ready!")
        print(f"📁 Output: {output_name}")
        print(f"📊 Video duration: {video_duration_ms / 1000:.0f}s")
        print(f"📝 Segments: {len(translated_segments)}")
        print(f"⏱️ Processing time: {minutes}m {seconds}s")
        print("=" * 55)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
import yt_dlp
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import subprocess
import os


# Step 1: Download YouTube video
def download_video(url):
    print("📥 Downloading video...")

    # Remove existing file to avoid conflicts
    if os.path.exists("video.mp4"):
        os.remove("video.mp4")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'video.mp4',
        'merge_output_format': 'mp4',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print("✅ Video downloaded successfully!")


# Step 2: Extract audio from video
def extract_audio():
    print("🔊 Extracting audio...")

    command = [
        "ffmpeg", "-y",
        "-i", "video.mp4",
        "-ar", "16000",
        "-ac", "1",
        "audio.wav"
    ]
    subprocess.run(command, check=True)

    print("✅ Audio extracted successfully!")


# Step 3: Speech to text using Whisper (auto-detects language)
def speech_to_text():
    print("🗣️ Converting speech to text (this may take a while)...")

    model = whisper.load_model("base")

    # First, detect the language
    print("   Detecting language...")
    audio = whisper.load_audio("audio.wav")
    audio_segment = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    confidence = probs[detected_lang]
    print(f"   Detected language: {detected_lang} (confidence: {confidence:.1%})")

    # Transcribe with detected language
    result = model.transcribe("audio.wav", language=detected_lang)
    text = result["text"]

    with open("original_text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print(f"✅ Speech converted to text ({len(text)} characters)")
    print(f"📝 Original text saved to original_text.txt")

    return text, detected_lang


# Step 4: Translate text to target language
def translate_text(text, source_lang, target_lang):
    print(f"🌐 Translating from '{source_lang}' to '{target_lang}'...")

    # If source and target are the same, skip translation
    if source_lang == target_lang:
        print("⚠️ Source and target language are the same. Skipping translation.")
        return text

    # Google Translator has a character limit, so split long texts
    max_chars = 4500
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"   Translating chunk {i + 1}/{len(chunks)}...")
        try:
            translated = GoogleTranslator(source=source_lang, target=target_lang).translate(chunk)
            if translated:
                translated_chunks.append(translated)
            else:
                print(f"   ⚠️ Chunk {i + 1} returned empty, using original")
                translated_chunks.append(chunk)
        except Exception as e:
            print(f"   ⚠️ Chunk {i + 1} failed: {e}")
            print(f"   Using 'auto' source detection for this chunk...")
            try:
                translated = GoogleTranslator(source='auto', target=target_lang).translate(chunk)
                translated_chunks.append(translated if translated else chunk)
            except Exception as e2:
                print(f"   ❌ Retry also failed: {e2}. Using original text.")
                translated_chunks.append(chunk)

    translated_text = " ".join(translated_chunks)

    with open("translated.txt", "w", encoding="utf-8") as f:
        f.write(translated_text)

    print(f"✅ Translation complete ({len(translated_text)} characters)")
    print(f"📝 Translated text saved to translated.txt")

    return translated_text


# Step 5: Generate audio from translated text
def generate_audio(text, language):
    print("🔈 Generating translated audio...")

    # gTTS language codes
    tts_lang_map = {
        'en': 'en',
        'te': 'te',
        'hi': 'hi',
    }

    tts_lang = tts_lang_map.get(language, language)
    tts = gTTS(text=text, lang=tts_lang)
    tts.save("translated_audio.mp3")

    print("✅ Translated audio generated!")


# Step 6: Merge translated audio with original video
def merge_audio_video():
    print("🎬 Merging audio with video...")

    command = [
        "ffmpeg", "-y",
        "-i", "video.mp4",
        "-i", "translated_audio.mp3",
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-shortest",
        "output.mp4"
    ]
    subprocess.run(command, check=True)

    print("✅ Final video created: output.mp4")


# Cleanup temporary files
def cleanup():
    print("🧹 Cleaning up temporary files...")
    for f in ["audio.wav", "translated_audio.mp3"]:
        if os.path.exists(f):
            os.remove(f)
    print("✅ Cleanup done!")


# Map Whisper language codes to Google Translate codes
WHISPER_TO_GOOGLE = {
    'en': 'en',
    'hi': 'hi',
    'te': 'te',
    'ta': 'ta',
    'mr': 'mr',
    'bn': 'bn',
    'gu': 'gu',
    'kn': 'kn',
    'ml': 'ml',
    'pa': 'pa',
    'ur': 'ur',
}


# Main function
def main():
    print("=" * 50)
    print("   🎥 VIDEO TRANSLATOR 🎥")
    print("   Auto-detect → English / Telugu / Hindi")
    print("=" * 50)
    print()

    url = input("Enter YouTube video URL: ").strip()
    if not url:
        print("❌ No URL provided. Exiting.")
        return

    print()
    print("Choose target language:")
    print("  1. English")
    print("  2. Telugu")
    print("  3. Hindi")
    print()

    choice = input("Enter choice (1, 2, or 3): ").strip()

    lang_map = {
        "1": ("en", "English"),
        "2": ("te", "Telugu"),
        "3": ("hi", "Hindi"),
    }

    if choice not in lang_map:
        print("❌ Invalid choice. Exiting.")
        return

    target, lang_name = lang_map[choice]

    print()
    print(f"🚀 Starting translation → {lang_name}")
    print("-" * 50)

    try:
        download_video(url)
        extract_audio()

        original_text, detected_lang = speech_to_text()
        print(f"🔍 Video language detected as: {detected_lang}")

        # Map detected language for Google Translate
        source_for_google = WHISPER_TO_GOOGLE.get(detected_lang, 'auto')
        print(f"   Using source language: {source_for_google}")

        translated_text = translate_text(original_text, source_for_google, target)
        generate_audio(translated_text, target)
        merge_audio_video()
        cleanup()

        print()
        print("=" * 50)
        print(f"🎉 DONE! Your {lang_name} video is ready!")
        print(f"📁 Output file: output.mp4")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
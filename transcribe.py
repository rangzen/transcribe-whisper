
import whisper
import argparse
import os

AUDIO_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".webm", ".mkv", ".avi", ".mov"}

def transcribe_audio(file_path, model_name, model=None):
    """
    Transcribes an audio file using OpenAI Whisper.

    Args:
        file_path (str): The path to the audio file.
        model_name (str): The name of the Whisper model to use.
        model: An already-loaded Whisper model (optional, avoids reloading).
    """
    if model is None:
        print(f"Loading Whisper model '{model_name}'...")
        model = whisper.load_model(model_name)

    print(f"Transcribing {file_path}...")
    result = model.transcribe(file_path, verbose=True)

    base_name = os.path.splitext(file_path)[0]
    output_path = base_name + ".txt"

    with open(output_path, "w") as f:
        f.write(result["text"])

    print(f"Transcription saved to {output_path}")

def transcribe_directory(dir_path, model_name):
    audio_files = [
        os.path.join(dir_path, f)
        for f in sorted(os.listdir(dir_path))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]

    if not audio_files:
        print(f"No audio files found in {dir_path}")
        return

    pending = [
        f for f in audio_files
        if not os.path.exists(os.path.splitext(f)[0] + ".txt")
    ]

    skipped = len(audio_files) - len(pending)
    if skipped:
        print(f"Skipping {skipped} already-transcribed file(s).")

    if not pending:
        print("All audio files already transcribed.")
        return

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    for file_path in pending:
        transcribe_audio(file_path, model_name, model=model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file or directory using OpenAI Whisper.")
    parser.add_argument("file", type=str, help="Path to an audio file or a directory of audio files.")
    parser.add_argument("--model", type=str, default="base", help="The name of the Whisper model to use (e.g., tiny, base, small, medium, large).")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: Path not found at {args.file}")
    elif os.path.isdir(args.file):
        transcribe_directory(args.file, args.model)
    else:
        transcribe_audio(args.file, args.model)

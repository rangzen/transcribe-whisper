
import whisper
import argparse
import os
import json

AUDIO_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".webm", ".mkv", ".avi", ".mov"}

def get_hf_token(cli_token):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return cli_token or os.environ.get("HF_TOKEN")

def run_diarization(file_path, hf_token):
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("Error: pyannote.audio is required for diarization. Install with: uv sync --extra diarize")
        raise SystemExit(1)
    import torch

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))

    import subprocess
    import numpy as np
    import torch

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", file_path],
        capture_output=True, text=True, check=True
    )
    parts = probe.stdout.strip().split(",")
    sample_rate, channels = int(parts[0]), int(parts[1])

    raw = subprocess.run(
        ["ffmpeg", "-i", file_path, "-f", "f32le", "-acodec", "pcm_f32le",
         "-ar", str(sample_rate), "-ac", str(channels), "-"],
        capture_output=True, check=True
    )
    audio = np.frombuffer(raw.stdout, dtype=np.float32).reshape(-1, channels).T
    waveform = torch.from_numpy(audio.copy())
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    # pyannote >= 4.0 returns DiarizeOutput; extract the Annotation for itertracks
    return getattr(output, "speaker_diarization", output)

def assign_speakers(segments, diarization):
    labeled = []
    for segment in segments:
        mid = (segment["start"] + segment["end"]) / 2
        speaker = "UNKNOWN"
        for turn, _, label in diarization.itertracks(yield_label=True):
            if turn.start <= mid <= turn.end:
                speaker = label
                break
        labeled.append({"speaker": speaker, "text": segment["text"].strip()})
    return labeled

def format_diarized(labeled_segments):
    lines = []
    current_speaker = None
    buffer = []

    for seg in labeled_segments:
        if seg["speaker"] != current_speaker:
            if current_speaker is not None:
                lines.append(f"[{current_speaker}] {' '.join(buffer)}")
            current_speaker = seg["speaker"]
            buffer = [seg["text"]]
        else:
            buffer.append(seg["text"])

    if current_speaker is not None:
        lines.append(f"[{current_speaker}] {' '.join(buffer)}")

    return "\n".join(lines)

def transcribe_audio(file_path, model_name, model=None, diarize=False, hf_token=None):
    base_name = os.path.splitext(file_path)[0]
    cache_path = base_name + ".whisper.json"

    if os.path.exists(cache_path):
        print(f"Loading cached Whisper result from {cache_path}...")
        with open(cache_path) as f:
            result = json.load(f)
    else:
        if model is None:
            print(f"Loading Whisper model '{model_name}'...")
            model = whisper.load_model(model_name)

        print(f"Transcribing {file_path}...")
        result = model.transcribe(file_path, verbose=True)

        with open(cache_path, "w") as f:
            json.dump(result, f)

    if diarize:
        print("Running speaker diarization...")
        diarization = run_diarization(file_path, hf_token)
        text = format_diarized(assign_speakers(result["segments"], diarization))
    else:
        text = result["text"]

    output_path = base_name + ".txt"

    with open(output_path, "w") as f:
        f.write(text)

    print(f"Transcription saved to {output_path}")

def transcribe_directory(dir_path, model_name, diarize=False, hf_token=None):
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
        transcribe_audio(file_path, model_name, model=model, diarize=diarize, hf_token=hf_token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file or directory using OpenAI Whisper.")
    parser.add_argument("file", type=str, help="Path to an audio file or a directory of audio files.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model to use (tiny, base, small, medium, large).")
    parser.add_argument("--diarize", action="store_true", help="Label speakers in the output (requires pyannote.audio and HF_TOKEN).")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token (or set HF_TOKEN in .env).")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: Path not found at {args.file}")
        raise SystemExit(1)

    hf_token = None
    if args.diarize:
        hf_token = get_hf_token(args.hf_token)
        if not hf_token:
            print("Error: --diarize requires a HuggingFace token. Set HF_TOKEN in .env or pass --hf-token.")
            raise SystemExit(1)

    if os.path.isdir(args.file):
        transcribe_directory(args.file, args.model, diarize=args.diarize, hf_token=hf_token)
    else:
        transcribe_audio(args.file, args.model, diarize=args.diarize, hf_token=hf_token)

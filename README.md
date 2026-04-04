# Transcribe Audio with OpenAI Whisper

Small helper to transcribe audio files locally using [OpenAI Whisper](https://github.com/openai/whisper).

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Usage

```bash
# Transcribe a single file
uv run python transcribe.py <audio_file> [--model MODEL]

# Transcribe all audio files in a directory (skips already-transcribed files)
uv run python transcribe.py <directory> [--model MODEL]
```

The transcription is saved as a `.txt` file next to each input file.

### Models

| Model  | Size  | Notes                        |
|--------|-------|------------------------------|
| tiny   | ~39M  | Fastest, least accurate      |
| base   | ~74M  | Default                      |
| small  | ~244M |                              |
| medium | ~769M |                              |
| large  | ~1.5G | Most accurate, slowest       |

### Examples

```bash
# Single file
uv run python transcribe.py interview.mp3 --model small
# Output: interview.txt

# Whole directory (model loaded once, files processed sequentially)
uv run python transcribe.py ./recordings --model small
# Output: ./recordings/file1.txt, ./recordings/file2.txt, ...
```

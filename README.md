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
uv run python transcribe.py <audio_file> [--model MODEL]
```

The transcription is saved as a `.txt` file next to the input file.

### Models

| Model  | Size  | Notes                        |
|--------|-------|------------------------------|
| tiny   | ~39M  | Fastest, least accurate      |
| base   | ~74M  | Default                      |
| small  | ~244M |                              |
| medium | ~769M |                              |
| large  | ~1.5G | Most accurate, slowest       |

### Example

```bash
uv run python transcribe.py interview.mp3 --model small
# Output: interview.txt
```

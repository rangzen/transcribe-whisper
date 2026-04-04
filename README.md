# Transcribe Audio with OpenAI Whisper

Small helper to transcribe audio files locally using [OpenAI Whisper](https://github.com/openai/whisper). Optionally labels speakers via [pyannote.audio](https://github.com/pyannote/pyannote-audio).

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

For speaker diarization, install the extra dependencies:

```bash
uv sync --extra diarize
```

## Usage

```bash
# Transcribe a single file
uv run python transcribe.py <audio_file> [--model MODEL]

# Transcribe all audio files in a directory (skips already-transcribed files)
uv run python transcribe.py <directory> [--model MODEL]

# Transcribe with speaker labels
uv run python transcribe.py <audio_file> --diarize
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

# With speaker diarization
uv run python transcribe.py interview.mp3 --model small --diarize
# Output (interview.txt):
# [SPEAKER_00] Hello, how are you?
# [SPEAKER_01] I'm doing well, thanks.
```

## Speaker Diarization

Diarization identifies who is speaking at each point and labels the output accordingly. It requires a HuggingFace token and model access.

### Setup

1. Create a token at <https://huggingface.co/settings/tokens>
2. Accept model terms at <https://huggingface.co/pyannote/speaker-diarization-3.1>
3. Accept model terms at <https://huggingface.co/pyannote/segmentation-3.0>
4. Copy `.env.example` to `.env` and set your token:

```bash
cp .env.example .env
# then edit .env and fill in HF_TOKEN=your_token_here
```

The `.env` file is loaded automatically. Alternatively, pass it directly:

```bash
uv run python transcribe.py interview.mp3 --diarize --hf-token hf_xxxxx
```

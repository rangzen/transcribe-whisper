
import whisper
import argparse
import os

def transcribe_audio(file_path, model_name):
    """
    Transcribes an audio file using OpenAI Whisper.

    Args:
        file_path (str): The path to the audio file.
        model_name (str): The name of the Whisper model to use.
    """
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"Transcribing {file_path}...")
    result = model.transcribe(file_path, verbose=True)

    # Create the output file path
    base_name = os.path.splitext(file_path)[0]
    output_path = base_name + ".txt"

    # Save the transcription to a text file
    with open(output_path, "w") as f:
        f.write(result["text"])

    print(f"Transcription saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using OpenAI Whisper.")
    parser.add_argument("file", type=str, help="The path to the audio file to transcribe.")
    parser.add_argument("--model", type=str, default="base", help="The name of the Whisper model to use (e.g., tiny, base, small, medium, large).")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
    else:
        transcribe_audio(args.file, args.model)

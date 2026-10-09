"""Everything that touches EmbeddingGemma 2: loading it, and turning one file
(or one search query) into a vector of 768 numbers.

index.py and search.py both import from here, so the files and the queries are
guaranteed to go through exactly the same model and land in the same space.
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def kind_of(path: Path) -> str | None:
    """'image', 'audio', 'video' or 'text', from the file extension (None = ignore the file)."""
    for kind, extensions in CONFIG["extensions"].items():
        if path.suffix.lower() in extensions:
            return kind
    return None


def load_model(text_only: bool = False) -> SentenceTransformer:
    """The full model reads text, images, video and audio (~3.5 GB RAM on CPU).
    A text query only needs the text part (~1.7 GB): same weights, same vector space."""
    skip = {"vision_config": None, "audio_config": None} if text_only else {}
    return SentenceTransformer(CONFIG["model_id"], config_kwargs=skip)


def video_frames(path: Path, n: int) -> list[Image.Image]:
    """Grab n evenly spaced frames with the ffmpeg command line (no extra Python video libs)."""
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={n / duration},scale=640:-2",
                        "-frames:v", str(n), f"{tmp}/%02d.png"], check=True)
        return [Image.open(f).convert("RGB") for f in sorted(Path(tmp).glob("*.png"))]


def embed_file(model: SentenceTransformer, path: Path) -> np.ndarray:
    """One file -> one normalized 768-number vector. The file name is never shown to the model,
    only the pixels, the waveform or the text inside."""
    kind = kind_of(path)
    if kind == "text":
        vec = model.encode(path.read_text(encoding="utf-8"), prompt_name=CONFIG["document_prompt"])
    elif kind == "image":
        vec = model.encode({"image": str(path)})
    elif kind == "audio":
        wave, sr = sf.read(str(path), dtype="float32")
        if wave.ndim == 2:  # stereo -> mono
            wave = wave.mean(axis=1)
        vec = model.encode({"audio": {"array": wave, "sampling_rate": sr}})
    elif kind == "video":
        vec = model.encode({"video": video_frames(path, CONFIG["video_frames"])})
    else:
        raise ValueError(f"unsupported file: {path.name}")
    return vec / np.linalg.norm(vec)


def embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    """A search query gets the 'SearchQuery' prompt; stored text got the 'Document' prompt."""
    vec = model.encode(text, prompt_name=CONFIG["query_prompt"])
    return vec / np.linalg.norm(vec)

"""Step 2: search the indexed files with your own words, or with another file.

    python search.py "my dog at the beach"
    python search.py "festival of lights" --only image video
    python search.py --file my_phone/fri_sound_a_dog_barking.wav --only image
    python search.py "stuck in the rain" --dim 256
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from embed import CONFIG, ROOT, embed_file, embed_query, load_model

sys.stdout.reconfigure(encoding="utf-8")  # emoji on Windows consoles
ICON = {"image": "📷", "audio": "🔊", "video": "🎞", "text": "💬"}

ap = argparse.ArgumentParser()
ap.add_argument("query", nargs="?", help="what you remember, in plain words")
ap.add_argument("--file", help="search with a photo / sound / video / text file instead")
ap.add_argument("--only", nargs="+", choices=list(ICON), help="return only these kinds")
ap.add_argument("--dim", type=int, default=768, choices=[768, 512, 256, 128],
                help="use only the first N numbers of each vector (Matryoshka)")
ap.add_argument("--top", type=int, default=CONFIG["top_k"])
args = ap.parse_args()
if not (args.query or args.file):
    ap.error("give a query in quotes, or --file")

# 1. load the stored vectors (made by index.py)
index_dir = ROOT / CONFIG["index_dir"]
items = json.loads((index_dir / "items.json").read_text(encoding="utf-8"))
vectors = np.load(index_dir / "vectors.npy")

# 2. turn the query into 768 numbers with the same model
if args.file:
    model = load_model()                       # a file needs the full model
    query = embed_file(model, Path(args.file))
else:
    model = load_model(text_only=True)         # words only need the text part
    query = embed_query(model, args.query)

# 3. compare: cosine similarity = dot product of normalized vectors
d = args.dim
stored = vectors[:, :d] / np.linalg.norm(vectors[:, :d], axis=1, keepdims=True)
scores = stored @ (query[:d] / np.linalg.norm(query[:d]))

# 4. rank and print
print(f'\n{args.query or args.file}   ({d} numbers per vector)\n')
shown = 0
for i in np.argsort(-scores):
    it = items[i]
    if args.only and it["kind"] not in args.only:
        continue
    if args.file and it["file"] == Path(args.file).name:
        continue                               # don't return the file you searched with
    print(f"  {scores[i]:.3f}  {ICON[it['kind']]} {it['kind']:5s}  {it['file']}")
    shown += 1
    if shown == args.top:
        break

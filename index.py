"""Step 1: embed every file in my_phone/ and store the vectors in index/.

    python index.py

Run it again after adding files: only new or changed files are embedded.
"""
import json
import time

import numpy as np

from embed import CONFIG, ROOT, embed_file, kind_of, load_model

files_dir = ROOT / CONFIG["files_dir"]
index_dir = ROOT / CONFIG["index_dir"]
items_path, vectors_path = index_dir / "items.json", index_dir / "vectors.npy"

# 1. what is already indexed? (a file counts as unchanged if its size and date are the same)
items, vectors = [], np.zeros((0, 768), dtype=np.float32)
if items_path.exists():
    items = json.loads(items_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path)
known = {(it["file"], it["size"], it["mtime"]) for it in items}

# 2. find new or changed files
files = sorted(p for p in files_dir.iterdir() if p.is_file() and kind_of(p))
todo = [p for p in files if (p.name, p.stat().st_size, int(p.stat().st_mtime)) not in known]
names_now = {p.name for p in files}
keep = [i for i, it in enumerate(items) if it["file"] in names_now and it["file"] not in {p.name for p in todo}]
items, vectors = [items[i] for i in keep], vectors[keep]
print(f"{len(files)} files in {files_dir.name}/, {len(items)} already indexed, {len(todo)} to embed")

# 3. embed them with the full model (text + image + audio + video)
if todo:
    model = load_model()
    new_vectors = []
    for n, path in enumerate(todo, 1):
        t0 = time.time()
        new_vectors.append(embed_file(model, path))
        items.append({"file": path.name, "kind": kind_of(path),
                      "size": path.stat().st_size, "mtime": int(path.stat().st_mtime)})
        print(f"  [{n}/{len(todo)}] {kind_of(path):5s} {path.name}  ({time.time() - t0:.1f}s)")
    vectors = np.vstack([vectors, np.array(new_vectors, dtype=np.float32)])

# 4. save: one row of 768 numbers per file, in the same order as items.json
index_dir.mkdir(exist_ok=True)
np.save(vectors_path, vectors)
items_path.write_text(json.dumps(items, indent=1), encoding="utf-8")
print(f"saved {len(items)} vectors of {vectors.shape[1]} numbers -> {index_dir.name}/")

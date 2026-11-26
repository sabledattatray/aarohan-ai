import sqlite3
from pathlib import Path

from aarohan.generation.images.local_pipeline import LocalNeuralProvider

conn = sqlite3.connect('data/memory/aarohan.sqlite')
c = conn.cursor()
c.execute("SELECT filename, prompt, aspect_ratio FROM image_library ORDER BY created_at ASC")
rows = c.fetchall()

provider = LocalNeuralProvider()
img_dir = Path('artifacts/images')

for fn, prompt, ar in rows:
    file_path = img_dir / fn
    # Only regenerate if file exists and is relatively small (procedural was <300KB) or old
    if file_path.exists() and file_path.stat().st_size < 350000:
        print(f"Regenerating procedural fallback: {fn} -> {prompt[:50]}")
        w, h = LocalNeuralProvider.ASPECT_RATIO_MAP.get(ar, (768, 768))
        try:
            img = provider._fetch_neural_image(prompt, w, h)
            if img:
                img.save(file_path, format="PNG", quality=95)
                print(f"  [OK] Saved {file_path} (new size: {file_path.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"  [ERR] {e}")

print("Upgrade complete!")

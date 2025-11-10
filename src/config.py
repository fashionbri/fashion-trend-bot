from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "history"      # your "raw"
LATEST = DATA / "latest"        # your "processed"
IMAGES = DATA / "images"
MODELS = ROOT / "models"

# make sure directories exist
for p in (DATA, HISTORY, LATEST, IMAGES, MODELS):
    p.mkdir(parents=True, exist_ok=True)

# env vars (for keys, etc.)
SERP_API_KEY = os.getenv("SERP_API_KEY", "")  # set in GitHub Actions secrets

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "history"      # this matches your folder
LATEST = DATA / "latest"        # this matches your folder
IMAGES = DATA / "images"
MODELS = ROOT / "models"

# create folders automatically
for p in (DATA, HISTORY, LATEST, IMAGES, MODELS):
    p.mkdir(parents=True, exist_ok=True)

# keep backward compatibility if scripts still call PROCESSED
PROCESSED = LATEST

# API keys
SERP_API_KEY = os.getenv("SERP_API_KEY", "")

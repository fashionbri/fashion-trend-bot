# 🕶️ Fashion Trend Bot

**Fashion Trend Bot** is an automated visual research tool that tracks emerging fashion trends by scanning runway and street-style imagery online twice a day.  
It uses AI-powered image search and color analysis to detect recurring hues, textures, and moods from the latest collections, campaigns, and cultural visuals.

---

### 💡 What It Does
- Searches Google Images (via [SerpAPI](https://serpapi.com)) for curated fashion queries.  
- Downloads new images every morning and evening.  
- Extracts dominant colors using OpenCV and KMeans clustering.  
- Creates daily CSV summaries and archives them for historical trend analysis.  
- Runs automatically with GitHub Actions (no manual work needed!).

---

### 🧠 Example Queries
These are the default searches (editable inside `search_and_update.py`):
```python
QUERIES = [
  "vogue runway emerald dress",
  "leopard print runway 1990s",
  "Gucci Fall 2013 runway",
  "Chanel haute couture 2020",
  "Balmain burgundy runway 2025"
]

fashion-trend-bot/
│
├── search_and_update.py        # Main bot logic
├── requirements.txt            # Python dependencies
│
├── .github/
│   └── workflows/
│       └── scheduler.yml       # Runs the bot twice a day
│
└── data/
    ├── latest/                 # Latest daily output
    └── history/                # Archived snapshots


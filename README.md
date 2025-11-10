#  Fashion Trend Bot <333 (I'm Just A Girl)

**Fashion Trend Bot** is an automated visual research tool that tracks emerging fashion trends by scanning runway and street-style imagery online twice a day.  
It uses AI-powered image search and color analysis to detect recurring hues, textures, and moods from the latest collections, campaigns, and cultural visuals.

---

**What It Does (End-to-End)**
Builds the query set

Uses a curated list of forward-looking fashion queries (global fashion weeks; street style; runway details; campaigns/editorials; 2025–2026 look-ahead).

Auto-generates city/context/year combos across Paris, Milan, New York, London, Tokyo, Seoul, Copenhagen, Stockholm, Sydney, Los Angeles.

Deduplicates, shuffles per run, and limits to QUERIES (default 25) for coverage + variety.

Collects recent image URLs

Queries SerpAPI Google Images with a recency filter (RECENCY_MODE: past day d, week w [default], or month m).

Paginates up to target images per query (IMAGES_PER_QUERY, default 60).

Caps images per domain (default 5) to prevent over-representation; dedupes while preserving order.

Downloads images (parallel, resilient)

Streams bytes with retries, size guard (MIN_BYTES), and timeout (TIMEOUT).

Hash-names files, infers extension from Content-Type, and writes to data/latest/images/.

Logs every file + source URL + query to data/latest/download_manifest.csv.

Extracts dominant colors (perceptual)

Fixes EXIF orientation, converts to RGB, resizes to MAX_SIDE (default 768) for speed.

Converts pixels to CIELAB and sub-samples (SAMPLE_PIXELS, default 120k).

Runs k-means (OpenCV) with deterministic RNG to get K swatches (K_COLORS, default 5) + shares per image.

Writes data/latest/colors_opencv.csv with: image_path, rank, color_hex, share.

Buckets colors into families
Converts HEX → HSV and maps to families:
neutrals, pastels, neon, earth, green/blue, blue/purple, jewel, primary/jewel.
Writes data/latest/colors_families.csv with: image_path, rank, color_hex, share, color_family.
Builds daily summaries

Uses each image’s #1 swatch (rank == 1) to compute:
data/latest/top_colors_today.csv → columns: color_hex, count.
data/latest/top_color_families_today.csv → columns: color_family, count.
Runs optional analysis modules (if present)

If a modules/ package is available:
Always-on light modules: textures, fabrics, motifs, mood, geography (uses queries/URLs).
Heavy modules (only if RUN_HEAVY=1): garments, accessories, silhouettes, styling, lighting.
Each module must expose run(...) and writes its own CSVs to data/latest/.

Archives a clean snapshot

Copies the day’s core CSVs to data/history/<YYYY-MM-DD_HHMM_UTC>/ (UTC stamp).

Leaves data/latest/ as the current run.

Pantone matching - the pipeline can compute nearest Pantone for each extracted HEX (ΔE in LAB) and emit:
data/latest/colors_pantone.csv → image_path, rank, color_hex, share, pantone_code, pantone_name, pantone_hex, deltaE
data/latest/top_pantone_today.csv → tally of nearest Pantone from each image’s #1 swatch.

import re, csv, sys
from pathlib import Path
from bs4 import BeautifulSoup

# --------- vocab lists you can expand any time ----------
FIBRES = [
 "cotton","wool","polyester","silk","linen","viscose","rayon","modal","lyocell","tencel",
 "nylon","acrylic","cashmere","mohair","alpaca","angora","hemp","jute","ramie","acetate","triacetate","spandex","cupro","polyamide"
]
FABRICS = [
 "denim","chiffon","satin","velvet","tweed","tulle","organza","lace","crepe","corduroy","jersey","knit","ribbed",
 "poplin","gabardine","taffeta","brocade","seersucker","fleece","flannel","leather","suede",
 "pleated","ruched","smocked","sequined","metallic","sheer","mesh","neoprene","vinyl","plush","canvas","chambray"
]
LENGTHS = ["mini","midi","maxi","cropped","ankle-length","floor-length","knee-length"]
CUTS = [
 "a-line","bodycon","sheath","wrap","bias-cut","empire","fit-and-flare","peplum",
 "drop-waist","boxy","oversized","tailored","straight-leg","wide-leg","flare"
]
GARMENTS = [
 "dress","gown","blazer","trench","cardigan","sweater","hoodie","cargo pants","jeans",
 "skirt","shorts","trousers","shirt","tank","tee","camisole","corset","coat","jumpsuit"
]
STYLES = [
 "y2k","utilitarian","preppy","grunge","minimal","maximal","coquette","athleisure",
 "streetwear","quiet luxury","romantic","boho","punk","edgy","futuristic","retro","vintage"
]

VOCABS = {
 "fibre": FIBRES,
 "fabric": FABRICS,
 "length": LENGTHS,
 "cut": CUTS,
 "garment": GARMENTS,
 "style": STYLES
}

def clean_text(text):
    # collapse whitespace + lowercase
    text = re.sub(r"\s+", " ", text).lower()

    # UK ↔ US normalization (word-boundary safe)
    variants = {
        # core spelling
        "fibre": "fiber",
        "colour": "color",
        "grey": "gray",
        "jewellery": "jewelry",
        "organise": "organize",
        "organisation": "organization",
        "centre": "center",
        "metre": "meter",
        "theatre": "theater",
        "catalogue": "catalog",
        "dialogue": "dialog",  # sometimes in copy
        "analyse": "analyze",
        "paralyse": "paralyze",
        "behaviour": "behavior",
        "odour": "odor",
        "mould": "mold",
        "moulding": "molding",
        "vapour": "vapor",
        "armour": "armor",
        "rumour": "rumor",
        "travelling": "traveling",
        "traveller": "traveler",
        "pyjamas": "pajamas",
        "cheque": "check",  # pattern context

        # fashion nouns that differ by region (normalize to US)
        "trousers": "pants",
        "jumper": "sweater",
        "trainers": "sneakers",
        "waistcoat": "vest",
        "plimsolls": "sneakers",
        "dungarees": "overalls",
        "pinafore": "jumper dress",
        "mac": "raincoat",
        "mackintosh": "raincoat",
        "anorak": "parka",
        "shoelace": "shoelace",  # unify hyphen variants next lines
        "shoe-lace": "shoelace",
        "bootlace": "shoelace",
        "boot-lace": "shoelace",
        "hemline": "hemline",  # includes rare UK “haemline”, handle below
        "haemline": "hemline",

        # seasonal labelling (normalize to FW/SS)
        "autumn winter": "fall winter",
        "aw": "fw",  # collection codes like AW24 -> FW24; we’ll ensure boundaries below
        "spring summer": "spring summer",
        "ss": "ss",

        # minor style words used in copy
        "colourway": "colorway",
        "poloneck": "turtleneck",
        "roll-neck": "turtleneck",
        "rollneck": "turtleneck",
    }

    # apply replacements with word boundaries to avoid mangling longer words
    for src, dst in variants.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)

    # also unify hyphen/space variants for a few fashion terms
    hyphen_variants = {
        r"\ba line\b": "a-line",
        r"\bfit and flare\b": "fit-and-flare",
        r"\bbias cut\b": "bias-cut",
        r"\bwide leg\b": "wide-leg",
        r"\bstraight leg\b": "straight-leg",
        r"\bboot cut\b": "boot-cut",
        r"\bcrew neck\b": "crewneck",
        r"\bv neck\b": "v-neck",
        r"\bcolor block(ed)?\b": "color-block",
        r"\bsee through\b": "sheer",
    }
    for pattern, repl in hyphen_variants.items():
        text = re.sub(pattern, repl, text)

    # normalize collection codes like “AW24”, “aw23/24” → “FW24”
    text = re.sub(r"\baw(\s?-?\s?\d{2})", r"fw\1", text)

    return text

def load_text(path: Path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        soup = BeautifulSoup(raw, "lxml")
        for bad in soup(["script","style","noscript"]): bad.decompose()
        raw = soup.get_t_
def extract_folder(in_dir: str, out_csv: str):
    """
    Walk `in_dir`, read .txt/.html files, normalize text, and write a CSV with:
      filepath, has_fiber, has_fabric, has_length, has_cut, has_garment, has_style, matches
    `matches` is a semicolon-joined list like "fiber:cotton; fabric:denim; style:boho"
    """
    import os, re, csv
    from bs4 import BeautifulSoup
    from pathlib import Path

    # --- vocab (keep in sync with the lists near the top of this file) ---
    FIBRES  = VOCABS["fibre"]
    FABRICS = VOCABS["fabric"]
    LENGTHS = VOCABS["length"]
    CUTS    = VOCABS["cut"]
    GARMS   = VOCABS["garment"]
    STYLES  = VOCABS["style"]

    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.lower()

    def read_text(path: Path) -> str:
        if path.suffix.lower() in {".html", ".htm"}:
            html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            for s in soup(["script", "style", "noscript"]):
                s.decompose()
            return clean_text(soup.get_text(" "))
        else:
            return clean_text(path.read_text(encoding="utf-8", errors="ignore"))

    def any_in(text: str, words: list[str]) -> tuple[bool, list[str]]:
        found = []
        for w in words:
            # word-boundary match to avoid partials (e.g., "cottony")
            if re.search(rf"\b{re.escape(w)}\b", text):
                found.append(w)
        return (len(found) > 0, found)

    in_dir = Path(in_dir)
    rows = []
    for p in in_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".txt", ".html", ".htm"}:
            continue
        txt = read_text(p)

        flags = {}
        matches_list = []

        for label, vocab in [
            ("fiber",   FIBRES),
            ("fabric",  FABRICS),
            ("length",  LENGTHS),
            ("cut",     CUTS),
            ("garment", GARMS),
            ("style",   STYLES),
        ]:
            has, hits = any_in(txt, vocab)
            flags[label] = has
            if hits:
                matches_list.extend([f"{label}:{h}" for h in hits])

        rows.append({
            "filepath": str(p),
            "has_fiber":   int(flags["fiber"]),
            "has_fabric":  int(flags["fabric"]),
            "has_length":  int(flags["length"]),
            "has_cut":     int(flags["cut"]),
            "has_garment": int(flags["garment"]),
            "has_style":   int(flags["style"]),
            "matches": "; ".join(sorted(set(matches_list))),
        })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filepath","has_fiber","has_fabric","has_length",
                "has_cut","has_garment","has_style","matches"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


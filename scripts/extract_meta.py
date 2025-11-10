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

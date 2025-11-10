import re

KEYWORDS = [
    "sheer","lace","leather","denim","crochet","metallic","sequins","neon",
    "ballet flats","kitten heels","oversized blazer","logo","corset","fringe",
    "floral","plaid","check","tulle","satin","vinyl","cargo","utility"
]

def extract_keywords(text: str):
    tx = (text or "").lower()
    found = []
    for k in KEYWORDS:
        if re.search(rf"\b{re.escape(k)}\b", tx):
            found.append(k)
    return sorted(set(found))

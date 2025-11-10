# scripts/run_extract.py
import argparse
from scripts.extract_meta import extract_folder

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_dir",  default="data/articles", help="folder with .txt/.html")
    ap.add_argument("--out", dest="out_csv", default="data/metadata.csv", help="output CSV path")
    args = ap.parse_args()

    extract_folder(args.in_dir, args.out_csv)
    print(f"✅ wrote {args.out_csv}")


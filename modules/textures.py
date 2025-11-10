import pandas as pd, numpy as np, cv2, json
from pathlib import Path

FAMILIES = ["matte","glossy","metallic","sheer","fuzzy","quilted","ribbed","distressed"]

def _features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F).var()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_mean = float(hsv[...,1].mean())/255.0
    v_mean = float(hsv[...,2].mean())/255.0
    edges = cv2.Canny(gray, 50, 150).mean()
    return dict(lapl=lap, s=s_mean, v=v_mean, edges=edges)

def _classify(feat):
    if feat["lapl"] > 200 and feat["s"] > 0.4 and feat["v"] > 0.6: return "glossy"
    if feat["s"] < 0.1 and feat["v"] > 0.85:                      return "sheer"
    if feat["lapl"] > 260 and feat["s"] > 0.6:                    return "metallic"
    if feat["edges"] > 25 and feat["s"] < 0.25:                   return "matte"
    return "matte"

def run(paths, out_dir):
    rows=[]
    for p in paths:
        try:
            bgr = cv2.imread(p); img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            f=_features(img); label=_classify(f)
            rows.append({"image_path":p,"primary_label":label,"features":json.dumps(f)})
        except Exception: continue
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir)/"textures.csv", index=False)
    if not df.empty:
        agg = df.groupby("primary_label")["image_path"].count().reset_index(name="count")
        agg["percent"]=agg["count"]/agg["count"].sum()
        agg.to_csv(Path(out_dir)/"textures_summary.csv", index=False)

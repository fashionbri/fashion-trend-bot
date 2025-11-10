import pandas as pd, cv2, numpy as np
from pathlib import Path

def _shape_heuristics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    aspect = h / max(w,1)
    density = edges.mean()
    return aspect, density

def _classify(aspect, density):
    if aspect > 1.7 and density < 10: return "column"
    if aspect > 1.4 and density > 15: return "hourglass"
    if aspect < 1.1 and density > 12: return "boxy"
    if aspect > 1.8 and density > 18: return "tailored"
    if aspect < 1.0: return "oversized"
    return "A-line"

def run(paths, out_dir):
    rows=[]
    for p in paths:
        try:
            bgr=cv2.imread(p); img=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            a,d=_shape_heuristics(img)
            rows.append({"image_path":p,"primary_label":_classify(a,d),"aspect":a,"edge_density":d})
        except Exception: continue
    df=pd.DataFrame(rows); df.to_csv(Path(out_dir)/"silhouettes.csv", index=False)
    if not df.empty:
        agg=df.groupby("primary_label")["image_path"].count().reset_index(name="count")
        agg["percent"]=agg["count"]/agg["count"].sum()
        agg.to_csv(Path(out_dir)/"silhouettes_summary.csv", index=False)

import pandas as pd, cv2
from pathlib import Path

def _index(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,50,150).mean()
    idx = min(100.0, edges*1.2)
    label = "maximalist" if idx>30 else "minimalist"
    return idx, label

def run(paths, out_dir):
    rows=[]
    for p in paths:
        try:
            bgr=cv2.imread(p); img=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            idx,label=_index(img)
            rows.append({"image_path":p,"styling_index":idx,"primary_label":label})
        except Exception: continue
    df=pd.DataFrame(rows); df.to_csv(Path(out_dir)/"styling.csv", index=False)
    if not df.empty:
        agg=df["primary_label"].value_counts(normalize=False).rename_axis("label").reset_index(name="count")
        agg["percent"]=agg["count"]/agg["count"].sum()
        agg.to_csv(Path(out_dir)/"styling_summary.csv", index=False)

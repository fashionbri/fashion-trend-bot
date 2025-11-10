import pandas as pd
from pathlib import Path

MAP = {
    "paris":"France/Paris","milan":"Italy/Milan","london":"UK/London","new york":"USA/New York",
    "tokyo":"Japan/Tokyo","seoul":"Korea/Seoul","shanghai":"China/Shanghai","copenhagen":"Denmark/Copenhagen"
}

def run(download_manifest, out_dir):
    rows=[]
    for row in download_manifest:
        q = (row["query"] or "").lower()
        tag = None
        for k,v in MAP.items():
            if k in q: tag = v; break
        rows.append({"image_path":row["path"],"query":row["query"],"region":tag or "unknown"})
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir)/"geography.csv", index=False)
    agg = df.groupby("region")["image_path"].count().reset_index(name="count").sort_values("count", ascending=False)
    agg["percent"]=agg["count"]/agg["count"].sum()
    agg.to_csv(Path(out_dir)/"geography_summary.csv", index=False)

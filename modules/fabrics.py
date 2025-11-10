import torch, pandas as pd, json
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import timm

FABRICS = ["denim","leather","lace","knit","satin","silk","velvet","tweed","chiffon","tulle","wool","linen"]

def _setup():
    model = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True)
    model.eval()
    preprocess = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                            T.Normalize(mean=(0.4814,0.4578,0.4082), std=(0.2686,0.2613,0.2758))])
    tokenizer = timm.layers.tokenizer.clip_tokenize
    text_tokens = tokenizer([f"a close-up of {f} fabric" for f in FABRICS])
    text_embed = model.encode_text(text_tokens)
    return model, preprocess, text_embed

@torch.no_grad()
def run(paths, out_dir):
    model, prep, text_embed = _setup()
    rows=[]
    for p in paths:
        try:
            img = prep(Image.open(p).convert("RGB")).unsqueeze(0)
            img_embed = model.encode_image(img)
            sim = (img_embed @ text_embed.T).softmax(dim=1)[0]
            topk = torch.topk(sim, k=3)
            labels = [FABRICS[i] for i in topk.indices.tolist()]
            scores = [float(x) for x in topk.values.tolist()]
            rows.append({"image_path":p,"primary_label":labels[0],
                         "labels": ";".join(labels), "scores": json.dumps(scores)})
        except Exception: continue
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir)/"fabrics.csv", index=False)
    if not df.empty:
        agg = df.groupby("primary_label")["image_path"].count().reset_index(name="count")
        agg["percent"]=agg["count"]/agg["count"].sum()
        agg.to_csv(Path(out_dir)/"fabrics_summary.csv", index=False)

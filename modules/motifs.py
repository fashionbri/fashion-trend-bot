import torch, pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import timm

MOTIFS = ["floral","animal print","geometric","stripe","plaid","polka dots","logo","lace","paisley"]

def _setup():
    model = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True)
    model.eval()
    preprocess = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                            T.Normalize((0.4814,0.4578,0.4082),(0.2686,0.2613,0.2758))])
    tokenizer = timm.layers.tokenizer.clip_tokenize
    text = tokenizer([f"a fabric with {m} pattern" for m in MOTIFS])
    text_emb = model.encode_text(text)
    return model, preprocess, text_emb

@torch.no_grad()
def run(paths, out_dir):
    model, prep, text_emb = _setup()
    rows=[]
    for p in paths:
        try:
            img_emb = model.encode_image(prep(Image.open(p).convert("RGB")).unsqueeze(0))
            prob = (img_emb @ text_emb.T).softmax(dim=1)[0]
            top = int(torch.argmax(prob))
            rows.append({"image_path":p,"primary_label":MOTIFS[top],"score":float(prob[top])})
        except Exception: continue
    df=pd.DataFrame(rows); df.to_csv(Path(out_dir)/"motifs.csv", index=False)
    if not df.empty:
        agg=df.groupby("primary_label")["image_path"].count().reset_index(name="count")
        agg["percent"]=agg["count"]/agg["count"].sum()
        agg.to_csv(Path(out_dir)/"motifs_summary.csv", index=False)

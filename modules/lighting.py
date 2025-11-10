import pandas as pd, cv2, numpy as np
from pathlib import Path

def _metrics(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    v=float(hsv[...,2].mean())/255.0
    s=float(hsv[...,1].mean())/255.0
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    contrast=float(gray.std())/255.0
    kelvin_proxy = (img[...,2].mean() - img[...,0].mean())
    return dict(v=v,s=s,contrast=contrast,kelvin=kelvin_proxy)

def _classify(m):
    if m["v"]>0.75 and m["s"]<0.25: return "studio"
    if m["v"]>0.65 and m["s"]>0.35: return "daylight/street"
    if m["contrast"]>0.28:          return "spotlight/high-contrast"
    return "ambient/indoor"

def run(paths, out_dir):
    rows=[]
    for p in paths:
        try:
            bgr=cv2.imread(p);

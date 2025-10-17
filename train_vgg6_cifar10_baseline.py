
# Single-file baseline; mirrors Q1 scripts. (Kept concise.)
from __future__ import annotations
import argparse, json, os, random, math
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)

class RandomErasingSquare(nn.Module):
    def __init__(self,p=0.5,area_ratio=0.02,val=0.0): super().__init__(); self.p=p; self.area_ratio=area_ratio; self.val=val
    def forward(self,x):
        import random, math
        if random.random()>self.p: return x
        c,h,w=x.shape; area=h*w; a=max(1,int(self.area_ratio*area)); s=max(1,min(int(math.sqrt(a)),h,w))
        y=random.randint(0,h-s); x0=random.randint(0,w-s); x[:,y:y+s,x0:x0+s]=self.val; return x

class VGG6(nn.Module):
    def __init__(self): 
        super().__init__()
        def blk(i,o): 
            return nn.Sequential(nn.Conv2d(i,o,3,1,1,bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f=nn.Sequential(blk(3,64),blk(64,64),nn.MaxPool2d(2),blk(64,128),blk(128,128),nn.MaxPool2d(2),blk(128,256),blk(256,256),nn.MaxPool2d(2))
        self.h=nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256*4*4,256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256,10))
    def forward(self,x): return self.h(self.f(x))

def tfms(hf,crop,cut,jit):
    a=[]; used=[]
    if crop: a.append(T.RandomCrop(32,padding=4)); used.append("RandomCrop(32,4)")
    if hf:   a.append(T.RandomHorizontalFlip()); used.append("RandomHorizontalFlip")
    if jit:  a.append(T.ColorJitter(0.2,0.2,0.2,0.02)); used.append("ColorJitter")
    a.extend([T.ToTensor(), T.Normalize(MEAN,STD)])
    if cut: a.append(RandomErasingSquare()); used.append("RandomErasingSquare(~2%)")
    v=T.Compose([T.ToTensor(), T.Normalize(MEAN,STD)])
    return T.Compose(a), v, v, used

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",default="./data"); p.add_argument("--out_dir",default="./runs/baseline_single")
    p.add_argument("--epochs",type=int,default=60); p.add_argument("--batch_size",type=int,default=128)
    p.add_argument("--lr",type=float,default=0.1); p.add_argument("--momentum",type=float,default=0.9)
    p.add_argument("--weight_decay",type=float,default=5e-4); p.add_argument("--label_smoothing",type=float,default=0.0)
    p.add_argument("--amp",action="store_true"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--aug_hflip",action="store_true"); p.add_argument("--aug_crop",action="store_true")
    p.add_argument("--aug_cutout",action="store_true"); p.add_argument("--aug_jitter",action="store_true")
    a=p.parse_args()
    random.seed(a.seed); os.environ["PYTHONHASHSEED"]=str(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir,exist_ok=True)
    with open(os.path.join(a.out_dir,"config.json"),"w") as f: json.dump(vars(a),f,indent=2)
    tr,v,t,used=tfms(a.aug_hflip,a.aug_crop,a.aug_cutout,a.aug_jitter)
    print("Augs:",", ".join(used) if used else "None", " | Norm:", MEAN, STD)
    ds_tr=torchvision.datasets.CIFAR10(a.data_dir,train=True,download=True,transform=tr)
    g=torch.Generator().manual_seed(a.seed); n=int(0.9*len(ds_tr)); m=len(ds_tr)-n
    tr_ds,va_ds=torch.utils.data.random_split(ds_tr,[n,m],generator=g)
    ds_te=torchvision.datasets.CIFAR10(a.data_dir,train=False,download=True,transform=t)
    ltr=DataLoader(tr_ds,batch_size=a.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    lva=DataLoader(va_ds,batch_size=a.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    lte=DataLoader(ds_te,batch_size=a.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    model=VGG6().to(device)
    crit=nn.CrossEntropyLoss(label_smoothing=a.label_smoothing).to(device)
    opt=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=a.momentum,weight_decay=a.weight_decay,nesterov=True)
    scaler=torch.cuda.amp.GradScaler(enabled=a.amp)
    import math
    def sched(ep,eps): 
        w=min(5,max(1,a.epochs//10)); e=ep/eps
        if e<w: return (e+1)/w
        prog=(e-w)/max(1e-8,(a.epochs-w)); return 0.5*(1+math.cos(math.pi*min(1.0,prog)))
    lr_sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lambda s: sched(s,len(ltr)))
    import csv
    with open(os.path.join(a.out_dir,"metrics.csv"),"w",newline="") as f: csv.DictWriter(f,fieldnames=["epoch","lr","train_loss","train_acc","val_loss","val_acc"]).writeheader()
    best=0.0; best_path=os.path.join(a.out_dir,"best.pt")
    def acc(o,y): _,p=o.max(1); return (p.eq(y).sum().item()/y.size(0))
    for ep in range(1,a.epochs+1):
        model.train(); tl=ta=n=0; 
        for x,y in ltr:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            if a.amp:
                with torch.cuda.amp.autocast(): o=model(x); l=crit(o,y)
                scaler.scale(l).backward(); scaler.step(opt); scaler.update()
            else:
                o=model(x); l=crit(o,y); l.backward(); opt.step()
            b=y.size(0); tl+=l.item()*b; ta+=acc(o,y)*b; n+=b
        model.eval(); vl=va=n2=0
        with torch.no_grad():
            for x,y in lva:
                x=x.to(device); y=y.to(device); o=model(x); l=crit(o,y)
                b=y.size(0); vl+=l.item()*b; va+=acc(o,y)*b; n2+=b
        tl/=n; ta/=n; vl/=n2; va/=n2; lr_now=opt.param_groups[0]["lr"]
        with open(os.path.join(a.out_dir,"metrics.csv"),"a",newline="") as f: 
            csv.DictWriter(f,fieldnames=["epoch","lr","train_loss","train_acc","val_loss","val_acc"]).writerow({"epoch":ep,"lr":lr_now,"train_loss":tl,"train_acc":ta,"val_loss":vl,"val_acc":va})
        lr_sched.step()
        if va>best: best=va; torch.save({"model":model.state_dict(),"epoch":ep,"val_acc":best}, best_path)
        print(f"[{ep:03d}] lr={lr_now:.5f} tr_loss={tl:.4f} tr_acc={ta:.4f} va_loss={vl:.4f} va_acc={va:.4f}")
    model.load_state_dict(torch.load(best_path,map_location=device)["model"])
    with torch.no_grad():
        tloss=tacc=nn2=0
        for x,y in lte:
            x=x.to(device); y=y.to(device); o=model(x); l=crit(o,y); b=y.size(0); tloss+=l.item()*b; tacc+=acc(o,y)*b; nn2+=b
    tloss/=nn2; tacc/=nn2
    with open(os.path.join(a.out_dir,"final_test_metrics.json"),"w") as f: json.dump({"test_loss":tloss,"test_top1_acc":tacc},f,indent=2)

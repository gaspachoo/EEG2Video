import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import wandb
from models.encoders import ShallowNetEncoder, MLPEncoder_feat

class CombinedEEGDataset(Dataset):
    def __init__(self, eeg, features, labels, blocks):
        self.eeg = eeg; self.features = features
        self.labels = labels; self.blocks = blocks

    def __getitem__(self, idx):
        xr = self.eeg[idx]
        xf = self.features[idx]
        xr = (xr - xr.mean(dim=1, keepdim=True)) / (xr.std(dim=1, keepdim=True) + 1e-6)
        xf = (xf - xf.mean()) / (xf.std() + 1e-6)
        return xr, xf, self.labels[idx], self.blocks[idx]

    def __len__(self): return len(self.labels)

class GLMNet(nn.Module):
    def __init__(self, g_enc, l_enc, num_classes=40, emb_dim=128):
        super().__init__(); self.g=g_enc; self.l=l_enc
        in_dim = self.g.output_dim + self.l.output_dim
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self,xr,xf):
        return self.embedding(torch.cat([self.g(xr), self.l(xf.view(xf.size(0),-1))],dim=1))

def parse_args():
    p=argparse.ArgumentParser(); home=os.environ["HOME"]
    p.add_argument("--eeg_path",type=str,default=f"{home}/EEG2Video/data/EEG_500ms_sw/sub1_segmented.npz")
    p.add_argument("--feat_path",type=str,default=f"{home}/EEG2Video/data/DE_500ms_sw/sub1_features.npz")
    p.add_argument("--save_dir",type=str,default=f"{home}/EEG2Video/Gaspard_model/checkpoints/cv_glmnet")
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--n_epochs",type=int,default=60)
    p.add_argument("--save_every",type=int,default=20); p.add_argument("--no_early_stop",action="store_true")
    p.add_argument("--use_wandb",action="store_true"); return p.parse_args()

def train_glmnet(a):
    os.makedirs(a.save_dir,exist_ok=True); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[GLMNet] Training encoder on", dev, flush=True)
    eeg_npz,feat_npz=np.load(a.eeg_path),np.load(a.feat_path)
    eeg=torch.tensor(eeg_npz["eeg"],dtype=torch.float32)
    feats=torch.tensor(feat_npz["de" if "DE" in a.feat_path else "psd"],dtype=torch.float32)
    labels=torch.tensor(eeg_npz["labels"],dtype=torch.long); blocks=torch.tensor(eeg_npz["blocks"],dtype=torch.long)
    data=CombinedEEGDataset(eeg,feats,labels,blocks)

    for fold in range(7):
        print(f"\n🔁 Fold {fold}")
        run=None
        if a.use_wandb:
            rid_path=os.path.join(a.save_dir,f"wandb_id_fold{fold}.txt")
            if os.path.exists(rid_path): rid=open(rid_path).read().strip(); resume="must"
            else:
                rid=wandb.util.generate_id(); open(rid_path,"w").write(rid); resume=None
            run=wandb.init(project="eeg2video-glmnet-v1",name=f"cv_glmnet_fold{fold}",config=vars(a),id=rid,resume=resume)

        vb=(fold-1)%7
        tr=[i for i in range(len(data)) if data.blocks[i] not in [fold,vb]]; va=[i for i in range(len(data)) if data.blocks[i]==vb]; te=[i for i in range(len(data)) if data.blocks[i]==fold]
        dl_tr=DataLoader(Subset(data,tr),batch_size=256,shuffle=True,num_workers=4,pin_memory=True)
        dl_va=DataLoader(Subset(data,va),batch_size=256,shuffle=False,num_workers=4,pin_memory=True)
        dl_te=DataLoader(Subset(data,te),batch_size=256,shuffle=False,num_workers=4,pin_memory=True)

        g=ShallowNetEncoder(62,eeg.shape[-1]).to(dev)
        g.load_state_dict(torch.load(os.path.join(os.environ["HOME"],"EEG2Video/Gaspard_model/checkpoints/cv_shallownet",f"best_fold{fold}.pth"),map_location=dev)["encoder"])
        with torch.no_grad(): g.output_dim=g(torch.randn(2,62,eeg.shape[-1],device=dev)).shape[1]

        l=MLPEncoder_feat(input_dim=feats.shape[1] * feats.shape[2]).to(dev)
        l.load_state_dict(torch.load(os.path.join(os.environ["HOME"],"EEG2Video/Gaspard_model/checkpoints/cv_mlp_DE",f"best_fold{fold}.pt"),map_location=dev),strict=False)
        with torch.no_grad(): l.output_dim = l(torch.randn(2, feats.shape[1] * feats.shape[2]).to(dev)).shape[1]

        model=GLMNet(g,l).to(dev)
        optim=torch.optim.Adam(model.parameters(),lr=a.lr); crit=nn.CrossEntropyLoss()
        best,wait,p=0,0,25

        for ep in range(1, a.n_epochs + 1):
            model.train()
            corr = tot = 0
            running_loss = 0

            for xr, xf, y, _ in tqdm(dl_tr, desc=f"[Fold {fold}] Epoch {ep}/{a.n_epochs}"):
                xr, xf, y = xr.to(dev), xf.to(dev), y.to(dev)
                optim.zero_grad()
                out = model(xr, xf)
                loss = crit(out, y)
                loss.backward()
                optim.step()

                running_loss += loss.item() * y.size(0)
                corr += (out.argmax(1) == y).sum().item()
                tot += y.size(0)

            t_acc = corr / tot
            avg_loss = running_loss / tot

            # Validation
            model.eval()
            corr = tot = 0
            with torch.no_grad():
                for xr, xf, y, _ in dl_va:
                    xr, xf, y = xr.to(dev), xf.to(dev), y.to(dev)
                    pred = model(xr, xf).argmax(1)
                    corr += (pred == y).sum().item()
                    tot += y.size(0)
            v_acc = corr / tot

            if run:
                run.log({
                    "train/loss": avg_loss,
                    "train/acc": t_acc,
                    "val/acc": v_acc,
                    "epoch": ep
                })

            if v_acc > best:
                best, wait = v_acc, 0
                torch.save(model.state_dict(), os.path.join(a.save_dir, f"best_fold{fold}.pt"))
            else:
                wait += 1
                if not a.no_early_stop and wait >= p:
                    print("⏹️ Early stop")
                    break


        model.load_state_dict(torch.load(os.path.join(a.save_dir,f"best_fold{fold}.pt"))); model.eval(); corr=tot=0
        with torch.no_grad():
            for xr,xf,y,_ in dl_te:
                xr,xf=xr.to(dev),xf.to(dev); pred=model(xr,xf).argmax(1); corr+=(pred.cpu()==y).sum().item(); tot+=y.size(0)
        test_acc=corr/tot; print(f"📊 Fold {fold} | Test {test_acc:.3f}")
        if run: run.log({"test/acc":test_acc}); run.finish()

if __name__=="__main__": train_glmnet(parse_args())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image
import time
import numpy as np
import math
import wandb
from datasets import load_dataset
import os


#define the model
class DinoRegressionHeteroImages(nn.Module):
    def __init__(self, dino_model, hidden_dim=128, dropout=0.1, dino_dim=1024):
        super().__init__()
        self.dino = dino_model  # ViT backbone (pre‑trained Dinov2)
        for p in self.dino.parameters():
            p.requires_grad = False 

        # **KEEP THE SAME LAYER NAMES AS THE EMBEDDING‑ONLY MODEL**
        self.embedding_to_hidden = nn.Linear(dino_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.out_mu = nn.Linear(hidden_dim, 1)
        self.out_logvar = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.dino(x)  # [B, dino_dim]
        h = self.embedding_to_hidden(h)
        h = self.leaky_relu(h)
        h = self.dropout(h)
        h = self.hidden_to_hidden(h)
        h = self.leaky_relu(h)
        mu = self.out_mu(h).squeeze(1)
        logvar = self.out_logvar(h).squeeze(1)
        logvar = torch.clamp(logvar, -10.0, 3.0)  # σ ~ [0.005, 20]
        return mu, logvar


# Standard image transform
imgtransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.convert('RGB')),  # Ensure images are in RGB format
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# This uses the Huggingface dataset library to load the dataset. 
class LifespanDataset(Dataset):
    def __init__(self, split="train",
                 repo_id="TristanKE/RemainingLifespanPredictionFaces",
                 transform=None):
        self.ds = load_dataset(repo_id, split=split)
        self.transform = transform

        remaining = np.array(self.ds["remaining_lifespan"], dtype=np.float32)
        self.lifespan_mean = float(remaining.mean())
        self.lifespan_std  = float(remaining.std())

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex  = self.ds[idx]                  # dict with keys: image, remaining_lifespan, …
        img = ex["image"]                   # PIL.Image
        if self.transform:
            img = self.transform(img)

        target = (ex["remaining_lifespan"] - self.lifespan_mean) / self.lifespan_std
        return img, torch.tensor(target, dtype=torch.float32)


# Gaussian Negative Log Likelihood loss 
def heteroscedastic_nll(y, mu, logvar):
    inv_var = torch.exp(-logvar)
    return (0.5 * inv_var * (y - mu) ** 2 + 0.5 * logvar).mean()


# Cosine learning rate scheduler
def cosine_schedule(epoch, total_epochs):
    return 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


# Main training loop
if __name__ == "__main__":
    # Configuration, here you can change most things including the dataset
    cfg = {
        "N_HEADONLY_EPOCHS": 0,
        "N_EPOCHS": 10,
        "BASE_LR": 1e-4,
        "BS": 32,
        "HIDDEN": 128,
        "DROPOUT": 0.01,
        "WANDB": True,
        "REPO_ID": "TristanKE/RemainingLifespanPredictionFaces",
        # "REPO_ID": "TristanKE/RemainingLifespanPredictionWholeImgs",
        "DINO_MODEL": "dinov2_vitl14_reg",
        # "DINO_MODEL": "dinov2_vitg14_reg", #the largest model, but also the slowest
        "DINO_DIM": 1024,
        # "DINO_DIM": 1536, #for the larger model
    }

    if cfg["WANDB"]:
        wandb.init(project="mortpred", config=cfg)

    torch.manual_seed(1)
    ds = LifespanDataset(repo_id=cfg["REPO_ID"],transform=imgtransform)

    test_sz = int(0.2 * len(ds))
    train_sz = len(ds) - test_sz
    train_ds, test_ds = random_split(ds, [train_sz, test_sz])

    train_dataset = Subset(ds, train_ds.indices)
    test_dataset = Subset(ds, test_ds.indices)
    train_loader = DataLoader(train_dataset, batch_size=cfg["BS"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg["BS"], shuffle=False, num_workers=4)

    # Load the model and move it to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_backbone = torch.hub.load("facebookresearch/dinov2", cfg["DINO_MODEL"]).to(device)
    model = DinoRegressionHeteroImages(dino_backbone, hidden_dim=cfg["HIDDEN"], dropout=cfg["DROPOUT"], dino_dim=cfg["DINO_DIM"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["BASE_LR"])
    scheduler = LambdaLR(optimizer, lambda e: cosine_schedule(e, cfg["N_EPOCHS"]))

    best_test_mae = float("inf")
    for epoch in range(cfg["N_EPOCHS"]):

        # Train
        model.train()
        tr_nll, tr_mae = 0.0, 0.0
        t0 = time.time()
        for imgs, tgt in train_loader:
            imgs, tgt = imgs.to(device), tgt.to(device)
            optimizer.zero_grad()
            mu, logvar = model(imgs)
            loss = heteroscedastic_nll(tgt, mu, logvar)
            loss.backward()
            optimizer.step()

            tr_nll += loss.item() * imgs.size(0)
            tr_mae += torch.abs(mu.detach() - tgt).sum().item()
            if cfg["WANDB"]:
                wandb.log({
                    "train_nll": loss.item(),
                    "train_mae": torch.abs(mu.detach() - tgt).mean().item() * ds.lifespan_std,
                    "train_std": torch.exp(0.5 * logvar).mean().item() * ds.lifespan_std,
                })

        tr_nll /= train_sz
        tr_mae = tr_mae / train_sz * ds.lifespan_std

        # Evaluate
        model.eval()
        te_nll, te_mae = 0.0, 0.0
        with torch.no_grad():
            for imgs, tgt in test_loader:
                imgs, tgt = imgs.to(device), tgt.to(device)
                mu, logvar = model(imgs)
                nll = heteroscedastic_nll(tgt, mu, logvar)
                te_nll += nll.item() * imgs.size(0)
                te_mae += torch.abs(mu - tgt).sum().item()
        te_nll /= test_sz
        te_mae = te_mae / test_sz * ds.lifespan_std

        print(f"Epoch {epoch+1}/{cfg['N_EPOCHS']} | {time.time()-t0:.1f}s | NLL tr {tr_nll:.3f} / te {te_nll:.3f} | MAE(te) {te_mae:.2f} yrs")

        if cfg["WANDB"]:
            wandb.log({
                "train_nll": tr_nll,
                "test_nll": te_nll,
                "test_mae_yrs": te_mae,
                "lr": scheduler.get_last_lr()[0],
            })

        scheduler.step()

        # save best
        if te_mae < best_test_mae:
            best_test_mae = te_mae
            if not os.path.exists("savedmodels"):
                os.makedirs("savedmodels")
            torch.save(model.state_dict(), f"savedmodels/dino_finetuned_faces_l1_{cfg['DINO_DIM']}_best.pth")
            print(f"\tNew best model saved (test MAE {te_mae:.3f})")

    if cfg["WANDB"]:
        wandb.finish()

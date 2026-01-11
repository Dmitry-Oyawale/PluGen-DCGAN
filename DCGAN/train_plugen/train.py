import os
import gin
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.flow import FlowModel
from src.data_utils import get_dataset
from src.utils import save_image_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# DCGAN Generator 
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        # z: [B, nz] or [B, nz, 1, 1]
        if z.dim() == 2:
            z = z[:, :, None, None]
        return self.main(z)


# Attribute classifier (multi-label, binary)
class AttrClassifier(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        # Simple CNN for 64x64
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1), # 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), # 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_features),
        )

    def forward(self, x):
        return self.head(self.features(x))


def train_classifier(clf, loader, num_epochs, lr, save_dir, num_features):
    clf.train()
    opt = optim.Adam(clf.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, num_epochs + 1):
        total = 0.0
        count = 0

        for imgs, attrs in tqdm(loader, desc=f"[clf] epoch {epoch}/{num_epochs}"):
            imgs = imgs.to(device)
            attrs = attrs.to(device)

            opt.zero_grad()
            logits = clf(imgs)
            loss = bce(logits, attrs)
            loss.backward()
            opt.step()

            total += loss.item() * imgs.size(0)
            count += imgs.size(0)

        avg = total / max(count, 1)
        print(f"[clf] epoch {epoch} loss={avg:.4f}")

        # Save checkpoint
        ckpt = os.path.join(save_dir, f"classifier_epoch_{epoch}.pt")
        torch.save(clf.state_dict(), ckpt)


def train_flow(flow, G, clf, loader, num_epochs, lr, nz, save_dir, sample_dir, lambda_z, samples_per_epoch):
    # Freeze G and classifier
    G.eval()
    clf.eval()
    for p in G.parameters():
        p.requires_grad = False
    for p in clf.parameters():
        p.requires_grad = False

    flow.train()
    opt = optim.Adam(flow.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, num_epochs + 1):
        total = 0.0
        count = 0

        for imgs_real, attrs_real in tqdm(loader, desc=f"[flow] epoch {epoch}/{num_epochs}"):
            # We DON'T use imgs_real directly to infer z (no encoder).
            # We only use attrs_real as target attribute vectors.
            attrs = attrs_real.to(device)

            # Sample z
            bsz = attrs.size(0)
            z = torch.randn(bsz, nz, device=device)

            opt.zero_grad()

            # Apply conditional flow: z' = f(z, attrs)
            z2 = flow(z, attrs)

            # Generate images from edited latent
            x2 = G(z2)

            # Attribute loss: classifier should predict attrs on generated images
            logits = clf(x2)
            loss_attr = bce(logits, attrs)

            # Regularize: keep edits small in latent space
            loss_reg = ((z2 - z) ** 2).mean()

            loss = loss_attr + lambda_z * loss_reg
            loss.backward()
            opt.step()

            total += loss.item() * bsz
            count += bsz

        avg = total / max(count, 1)
        print(f"[flow] epoch {epoch} loss={avg:.4f}")

        # Save samples each epoch
        with torch.no_grad():
            k = min(samples_per_epoch, 64)
            z = torch.randn(k, nz, device=device)

            # pick some attribute vectors from last seen batch (or random zeros if empty)
            if 'attrs' in locals() and attrs.size(0) > 0:
                a = attrs[:k]
                if a.size(0) < k:
                    # pad by repeating
                    reps = (k + a.size(0) - 1) // a.size(0)
                    a = a.repeat(reps, 1)[:k]
            else:
                a = torch.zeros(k, loader.dataset.num_features, device=device)

            z2 = flow(z, a)
            x = G(z)   # baseline
            x2 = G(z2) # edited
            save_image_grid(x, os.path.join(sample_dir, f"baseline_epoch_{epoch}.png"))
            save_image_grid(x2, os.path.join(sample_dir, f"edited_epoch_{epoch}.png"))

        # Save flow checkpoint
        ckpt = os.path.join(save_dir, f"flow_epoch_{epoch}.pt")
        torch.save(flow.state_dict(), ckpt)


@gin.configurable
def train(
    # data
    image_dir,
    # model
    num_features,
    nz,
    ngf,
    generator_checkpoint,
    flow_fn,
    # classifier
    clf_epochs,
    clf_lr,
    # flow training
    flow_epochs,
    flow_lr,
    lambda_z,
    batch_size,
    num_workers,
    # misc
    model_name,
    samples_per_epoch,
):
    # Save paths
    root = Path("saved") / model_name
    root.mkdir(parents=True, exist_ok=True)

    preds_dir = Path("preds") / model_name
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Dataset + loader (images + binary attrs from caption txt)
    dataset = get_dataset(image_dir=image_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    # Load pretrained generator
    G = Generator(nz=nz, ngf=ngf, nc=3).to(device)
    if generator_checkpoint and os.path.exists(generator_checkpoint):
        G.load_state_dict(torch.load(generator_checkpoint, map_location=device))
        print(f"Loaded generator checkpoint: {generator_checkpoint}")
    else:
        raise FileNotFoundError(f"Generator checkpoint not found: {generator_checkpoint}")

    # Build classifier
    clf = AttrClassifier(num_features=num_features).to(device)

    # Train classifier on real images
    print("==> Training attribute classifier on real images...")
    train_classifier(
        clf=clf,
        loader=loader,
        num_epochs=clf_epochs,
        lr=clf_lr,
        save_dir=str(root),
        num_features=num_features,
    )

    # Freeze classifier and train flow
    flow = flow_fn().to(device)

    print("==> Training conditional flow (PluGeN) on DCGAN latent...")
    train_flow(
        flow=flow,
        G=G,
        clf=clf,
        loader=loader,
        num_epochs=flow_epochs,
        lr=flow_lr,
        nz=nz,
        save_dir=str(root),
        sample_dir=str(preds_dir),
        lambda_z=lambda_z,
        samples_per_epoch=samples_per_epoch,
    )

    # Save final artifacts
    torch.save(flow.state_dict(), str(root / "final_flow.pt"))
    torch.save(clf.state_dict(), str(root / "final_classifier.pt"))
    print(f"Saved final models to: {root}")


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="configs/config.gin")

if __name__ == "__main__":
    args = parser.parse_args()
    gin.parse_config_file(args.config_file)
    train()

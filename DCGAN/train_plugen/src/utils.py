import os
import re
import gin
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


TAGS: List[str] = [
    "has_helmet",
    "is_colorful",
    "is_dark",
    "is_female",
    "is_robot",
    "is_animal_themed",
]

PATTERNS: Dict[str, List[str]] = {
    "has_helmet": [
        r"\bhelmet\b",
        r"\bheadgear\b",
        r"\bvisor\b",
        r"\bface\s*mask\b",
        r"\bmask\b",
        r"\bgas\s*mask\b",
    ],
    "is_robot": [
        r"\brobot\b",
        r"\bandroid\b",
        r"\bcyborg\b",
        r"\bmech\b",
        r"\bmechanical\b",
        r"\bmetal(?:lic)?\b",
        r"\bmachine\b",
    ],
    "is_animal_themed": [
        r"\banimal\b",
        r"\bcat\b",
        r"\bdog\b",
        r"\bwolf\b",
        r"\bfox\b",
        r"\bbear\b",
        r"\brabbit\b",
        r"\bbunny\b",
        r"\btiger\b",
        r"\blion\b",
        r"\bpanda\b",
        r"\bbird\b",
        r"\bowl\b",
        r"\bdragon\b",
        r"\bdinosaur\b",
        r"\bdeer\b",
        r"\bshark\b",
        r"\bfrog\b",
    ],
    "is_female": [
        r"\bfemale\b",
        r"\bwoman\b",
        r"\bgirl\b",
        r"\blady\b",
        r"\bprincess\b",
    ],
    "is_dark": [
        r"\bdark\b",
        r"\bblack\b",
        r"\bshadow\b",
        r"\bnight\b",
        r"\bgoth\b",
        r"\bemo\b",
        r"\bcharcoal\b",
        r"\bmidnight\b",
    ],
    "is_colorful": [
        r"\bcolorful\b",
        r"\brainbow\b",
        r"\bmulticolor(?:ed)?\b",
        r"\bvibrant\b",
        r"\bbright\b",
        r"\bneon\b",
    ],
}

COMPILED: Dict[str, List[re.Pattern]] = {
    k: [re.compile(p) for p in v] for k, v in PATTERNS.items()
}


def caption_to_features(caption: str) -> torch.Tensor:
    c = (caption or "").lower()
    feats = []
    for tag in TAGS:
        regs = COMPILED.get(tag, [])
        val = 1.0 if any(r.search(c) for r in regs) else 0.0
        feats.append(val)
    return torch.tensor(feats, dtype=torch.float32)


class SkinsWithCaptionsDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int = 64):
        self.image_dir = image_dir
        self.image_size = image_size

        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Find png files that also have a .txt caption
        pngs = [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
        pairs: List[Tuple[str, str]] = []
        for png in pngs:
            base = os.path.splitext(png)[0]
            txt = base + ".txt"
            txt_path = os.path.join(image_dir, txt)
            if os.path.exists(txt_path):
                pairs.append((os.path.join(image_dir, png), txt_path))

        if not pairs:
            raise FileNotFoundError(
                f"No (png, txt) pairs found in {image_dir}. "
                f"Expected files like name.png and name.txt"
            )

        self.pairs = pairs
        self.num_features = len(TAGS)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, txt_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            caption = f.read().strip()

        feats = caption_to_features(caption) 
        return img, feats


@gin.configurable
def get_dataset(image_dir: str, image_size: int = 64):
    return SkinsWithCaptionsDataset(image_dir=image_dir, image_size=image_size)

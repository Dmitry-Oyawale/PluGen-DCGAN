import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as T


class ImageAttributeDataset(Dataset):
    def __init__(self, image_dir, csv_path, image_size=64):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5)),
        ])

        self.feature_cols = [
            c for c in self.df.columns
            if c.startswith("has_") or c.startswith("is_")
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row["file_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        attrs = torch.tensor(
            row[self.feature_cols].values,
            dtype=torch.float32
        )

        return image, attrs


def get_dataset(image_dir, csv_path, batch_size):
    dataset = ImageAttributeDataset(image_dir, csv_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    return loader, len(dataset.feature_cols)



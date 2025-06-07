import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToDtype
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset


class MyYoloDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = Compose(
            [
                ToDtype(torch.float32, scale=True),
            ]
        )

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out["img"] = self.transform(out["img"])
        return out


class YOLODataModule(LightningDataModule):
    def __init__(self, data_yaml, img_size=640, batch_size=16, num_workers=4):
        super().__init__()
        self.data_yaml = data_yaml
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Validate dataset config
        data_info = check_det_dataset(self.data_yaml, autodownload=False)

        # Create train dataset
        self.train_dataset = MyYoloDataset(
            img_path=data_info["train"],
            imgsz=self.img_size,
            augment=True,
            data=data_info,
        )

        # Create validation dataset
        self.val_dataset = MyYoloDataset(
            img_path=data_info["val"],
            imgsz=self.img_size,
            augment=False,
            data=data_info,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=YOLODataset.collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=YOLODataset.collate_fn,
            persistent_workers=True,
        )

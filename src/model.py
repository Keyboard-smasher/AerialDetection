import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss


class Mydict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box = None
        self.dfl = None
        self.cls = None


class V8DetectionLossModified(v8DetectionLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyp = Mydict(self.hyp)
        self.hyp.box = self.hyp["box"]
        self.hyp.cls = self.hyp["cls"]
        self.hyp.dfl = self.hyp["dfl"]


class YOLOLightning(LightningModule):
    def __init__(
        self,
        config_path: str,
        batch_size=16,
        lr: float = 0.01,
        img_size=640,
        save_onnx=True,
        save_tensorrt=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize YOLO model
        self.model = YOLO(config_path)
        self.grads = [i.requires_grad for i in self.model.parameters()]
        self.model.train = lambda *args: self.set_grads()
        self.model.eval = lambda *args: self.disable_grads()

        self.model.model.criterion = V8DetectionLossModified(self.model.model)

        # Loss names for logging
        self.loss_names = ["box_loss", "cls_loss", "dfl_loss"]

        self.metrics = {
            "MAP": MeanAveragePrecision(
                iou_type="bbox",
                box_format="xywh",
                iou_thresholds=[0.5, 0.75],
                class_metrics=True,
            ),
            "IOU": IntersectionOverUnion(box_format="xywh", iou_threshold=0.5, class_metrics=False),
        }
        self.lr = lr
        self.batch_size = batch_size
        self.img_size = img_size

        self.save_onnx = save_onnx
        self.save_tensorrt = save_tensorrt

        torch.set_float32_matmul_precision("medium")

    def to(self, device):
        self.model.to(device)
        self.model.model.criterion = V8DetectionLossModified(self.model.model)

    def forward(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def set_grads(self):
        for param, _ in zip(self.model.parameters(), self.grads):
            param.requires_grad = True

    def disable_grads(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        self.set_grads()

        with torch.enable_grad():
            # Compute loss
            loss, loss_items = self.model.loss(batch)
        # Log losses
        self.log(
            "train_loss",
            torch.sum(loss).item(),
            prog_bar=True,
            on_epoch=True,
            batch_size=16,
        )
        for name, item in zip(self.loss_names, loss_items):
            self.log(f"train_{name}", item, on_epoch=True, batch_size=self.batch_size)

        return torch.sum(loss)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, loss_items = self.model.loss(batch)

        # Log losses
        self.log(
            "val_loss",
            torch.sum(loss).item(),
            prog_bar=True,
            on_epoch=True,
            batch_size=16,
        )
        for name, item in zip(self.loss_names, loss_items):
            self.log(f"val_{name}", item, on_epoch=True, batch_size=self.batch_size)

    def on_validation_epoch_end(self):
        self.logger.save()

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-2,
        )

        # Learning rate scheduler
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150]),
            "interval": "epoch",
        }

        return [optimizer], [scheduler]

    def on_fit_end(self):
        self.save_model("last")
        self.process_metrics()

    def final_conversion(self):
        path_to_best = next(Path(self.logger.save_dir).rglob("best.ckpt"))

        checkpoint = torch.load(path_to_best, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])

        self.save_model("best")
        save_dir = self.return_save_folder()

        if self.save_tensorrt:
            self.export_yolo_model(save_dir / "best.engine", "engine")
            # self.model.export(format="engine", path=save_dir / "best.engine")

        if self.save_onnx:
            self.export_yolo_model(save_dir / "best.onnx", "onnx")
            # self.model.export(format="onnx", path=save_dir / "best.onnx")

    def calc_metrics(self, dataloader):
        print("Calculating metrics")
        metrics = []

        for data in dataloader:
            pred = self.model.predict(data["img"].to(self.device), verbose=False)

            processed_pred = self.prepare_pred(pred)
            processed_target = self.prepare_target(data)

            [metric.update(processed_pred, processed_target) for metric in self.metrics.values()]

            metric_vals = {}
            [metric_vals.update(metric.compute()) for _, metric in self.metrics.items()]
            metrics.append(metric_vals)

        out = pd.DataFrame(metrics)
        des_cols = ["map", "map_50", "map_75", "mar_1", "mar_10", "mar_100", "iou"]
        data = np.array(out[des_cols].values)

        data = dict(zip(des_cols, np.mean(data, axis=0)))
        print(pd.DataFrame(data, index=["Mean metrics"]))

    def return_save_folder(self):
        save_dir = Path(self.logger.save_dir) / "weights"
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        return save_dir

    def process_metrics(self):
        csv_path = next(Path(self.logger.save_dir).rglob("*.csv"))

        df = pd.read_csv(csv_path)
        df = df[[i for i in list(df.columns) if i.endswith("epoch") or i.endswith("loss")]]

        data = [df[df["epoch"] == i].iloc[-2:] for i in df["epoch"].unique()]

        cols = df.columns

        vals_coll = []

        for d_ in data:
            vals = np.zeros(cols.__len__())
            for i in range(2):
                tmp = d_.iloc[i].values
                vals[~np.isnan(tmp)] = tmp[~np.isnan(tmp)]
            vals_coll.append(vals)

        vals_coll = np.array(vals_coll)

        for i in range(1, len(cols)):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(vals_coll[:, 0], vals_coll[:, i])
            ax.grid(True)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_title(cols[i])
            fig.tight_layout()
            fig.savefig(csv_path.parent / (cols[i] + ".png"))

    def export_yolo_model(self, model_path, export_format):
        # Export with automatic naming
        temp_path = self.model.export(format=export_format, imgsz=self.img_size, nms=True)

        temp_path = Path(temp_path)

        shutil.move(temp_path, model_path)
        print(f"Model exported to {model_path}")

    def save_model(self, name):
        save_dir = self.return_save_folder()

        self.model.save(save_dir / f"{name}.pt")

    def prepare_target(self, sample):
        idx, bboxes, cls = (
            sample["batch_idx"],
            sample["bboxes"],
            sample["cls"].squeeze().int(),
        )
        elements = [idx == i for i in range(idx.max().int() + 1)]
        return [{"boxes": bboxes[e].to(self.device), "labels": cls[e].to(self.device)} for e in elements]

    def prepare_pred(self, out):
        elements = [elem.boxes for elem in out]
        return [
            {
                "boxes": e.xywh.to(self.device),
                "scores": e.conf.to(self.device),
                "labels": e.cls.int().to(self.device),
            }
            for e in elements
        ]

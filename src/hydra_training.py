from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.data import YOLODataModule
from src.model import YOLOLightning


# @hydra.main(config_path="../configs", config_name="main", version_base="1.3")
def train(cfg):
    OmegaConf.to_yaml(cfg)

    yolo_cfg = dict(cfg.model.model)
    yolo_cfg.update(
        {
            "batch_size": cfg.data.data.batch_size,
            "img_size": cfg.data.data.img_size,
            "lr": cfg.training.train_hyp.lr,
        }
    )

    print(yolo_cfg)

    # Initialize model and data
    model = YOLOLightning(**yolo_cfg)
    datamodule = YOLODataModule(**dict(cfg.data.data))

    # Initialize trainer
    trainer = Trainer(
        callbacks=[
            EarlyStopping(**dict(cfg.training.callback_stopping)),
            ModelCheckpoint(**dict(cfg.training.callback_checkpoint)),
        ],
        logger=[CSVLogger(**dict(cfg.training.logger))],
        **dict(cfg.training.trainer),
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)

    model.final_conversion()

    # Save final config
    OmegaConf.save(cfg, f"{trainer.logger.log_dir}/final_config.yaml")

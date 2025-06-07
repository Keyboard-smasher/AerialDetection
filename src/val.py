from ultralytics import YOLO


def val(cfg):
    model = YOLO(cfg.validation.validation.model_path)
    model.val(data=cfg.validation.validation.data_path)

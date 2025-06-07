import fire
import hydra
from hydra.core.hydra_config import HydraConfig

from src.hydra_training import train
from src.inference import inference
from src.val import val


def main(command: str) -> None:
    """
    Function to start any feature in the code
    :param command: name of the feature. Could be either 'train', 'val',
    'inference'
    """
    with hydra.initialize(config_path="configs", version_base="1.3", job_name="yolo_training"):
        cfg = hydra.compose(config_name="main", return_hydra_config=True)
        HydraConfig().set_config(cfg)

    if command == "train":
        train(cfg)

    elif command == "val":
        val(cfg)

    elif command == "inference":
        inference(cfg)

    else:
        raise ValueError(f"Command {command} not recognized")


if __name__ == "__main__":
    fire.Fire(main)

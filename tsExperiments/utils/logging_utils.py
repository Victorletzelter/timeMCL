from typing import Any, Dict, Optional

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

from typing import List, Dict, Any
from lightning.pytorch.loggers import Logger

@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any], logger_list: List[Logger]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])

    # trainer = object_dict["trainer"]

    # if not trainer.logger:
    #     log.warning("Logger not found! Skipping hyperparameter logging...")
    #     return

    model = object_dict["model"]
    hparams["model"] = cfg["model"]

    if model is not None:

        # save number of model parameters
        hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
        hparams["model/params/trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        hparams["model/params/non_trainable"] = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    for logger in logger_list:
        logger.log_hyperparams(hparams)

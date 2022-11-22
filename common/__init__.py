from hydra.utils import instantiate as hydra_inst
import os
from pytorch_lightning import LightningModule
from typing import Any

def set_cuda_devices_env(devices: list[str]):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	print(f"Set CUDA_DEVICE_ORDER to '{os.environ['CUDA_DEVICE_ORDER']}'")
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if not devices else ",".join(map(str, devices))
	print(f"Set CUDA_VISIBLE_DEVICES to '{os.environ['CUDA_VISIBLE_DEVICES']}'")

def load_model(config: Any) -> LightningModule:
	if config.get("checkpoint") is not None:
		print(f"Loading model from checkpoint: {config.checkpoint}")
		return hydra_inst(config.model, _target_=f"{config.model._target_}.load_from_checkpoint", checkpoint_path=config.checkpoint)
	return hydra_inst(config.model)


__all__ = [
	"load_model",
	"set_cuda_devices_env",
]
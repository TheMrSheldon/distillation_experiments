from hydra.utils import instantiate as hydra_inst
import os
from pathlib import Path
from pytorch_lightning import LightningModule
from typing import Any, Optional

def set_cuda_devices_env(devices: list[str]):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	print(f"Set CUDA_DEVICE_ORDER to '{os.environ['CUDA_DEVICE_ORDER']}'")
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if not devices else ",".join(map(str, devices))
	print(f"Set CUDA_VISIBLE_DEVICES to '{os.environ['CUDA_VISIBLE_DEVICES']}'")

def load_model(config: Any, checkpoint_path: Optional[Path] = None) -> LightningModule:
	if config.get("checkpoint") is not None:
		checkpoint = checkpoint_path / config.get("checkpoint") if checkpoint_path else Path(config.get("checkpoint"))
		print(f"Loading model from checkpoint: {checkpoint}")
		return hydra_inst(config.model, _target_=f"{config.model._target_}.load_from_checkpoint", checkpoint_path=checkpoint)
	return hydra_inst(config.model)


__all__ = [
	"load_model",
	"set_cuda_devices_env",
]
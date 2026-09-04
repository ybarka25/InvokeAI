from typing import Dict, Optional

import torch

from invokeai.backend.patches.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.patches.layers.param_shape_utils import get_param_shape
from invokeai.backend.util.calc_tensor_size import calc_tensor_size


class FullLayer(LoRALayerBase):
    """A full ("diff") weight/bias replacement layer.

    Some LyCORIS exports (e.g. bias/norm-only step-distillation LoRAs) only ship
    a diff for one of weight/bias, not both -- both are optional here.
    """

    def __init__(self, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]):
        super().__init__(alpha=None, bias=bias)
        self.weight = torch.nn.Parameter(weight) if weight is not None else None

    @classmethod
    def from_state_dict_values(
        cls,
        values: Dict[str, torch.Tensor],
    ):
        layer = cls(weight=values.get("diff", None), bias=values.get("diff_b", None))
        cls.warn_on_unhandled_keys(values=values, handled_keys={"diff", "diff_b"})
        return layer

    def _rank(self) -> int | None:
        return None

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        assert self.weight is not None
        return self.weight

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        if self.weight is None:
            # Bias/norm-only diff: no weight key to patch at all.
            params: dict[str, torch.Tensor] = {}
            bias = self.get_bias(orig_parameters.get("bias", None))
            if bias is not None:
                params["bias"] = bias * weight
                orig_bias = orig_parameters["bias"]
                if params["bias"].shape != get_param_shape(orig_bias):
                    params["bias"] = params["bias"].reshape(get_param_shape(orig_bias))
            return params
        return super().get_parameters(orig_parameters, weight)

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        if self.weight is not None:
            self.weight = self.weight.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        size = super().calc_size()
        if self.weight is not None:
            size += calc_tensor_size(self.weight)
        return size

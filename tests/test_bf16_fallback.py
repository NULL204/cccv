from typing import Any, Dict, Tuple

import torch
from torch import nn

from cccv.model.base_model import CCBaseModel


class _Bf16FailingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bfloat16:
            raise RuntimeError("bf16 kernel unavailable")
        return x + self.weight.to(x.dtype)


class _Bf16PreflightModel(CCBaseModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.load_count = 0
        super().__init__(*args, **kwargs)

    def load_model(self) -> nn.Module:
        self.load_count += 1
        return _Bf16FailingModule().eval().to(self.device)

    def get_bf16_preflight_inputs(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        img = torch.zeros((1, 1), device=self.device, dtype=self.half_dtype)
        return (img,), {}

    def inference(self, img: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return self.model(img)


def test_bf16_preflight_falls_back_to_fp32_when_fp16_disabled() -> None:
    model = _Bf16PreflightModel(config=object(), device=torch.device("cpu"), fp16=False, bf16=True)

    assert model.load_count == 2
    assert model.bf16 is False
    assert model.fp16 is False
    assert model.half_dtype == torch.float32
    assert next(model.model.parameters()).dtype == torch.float32


def test_bf16_preflight_falls_back_to_fp16_when_requested() -> None:
    model = _Bf16PreflightModel(config=object(), device=torch.device("cpu"), fp16=True, bf16=True)

    assert model.load_count == 2
    assert model.bf16 is False
    assert model.fp16 is True
    assert model.half_dtype == torch.float16
    assert next(model.model.parameters()).dtype == torch.float16

#!/usr/bin/env python3
"""Expert ReLU MLP with explicit forward/backward epilogue units."""

import argparse
from dataclasses import dataclass

import torch
from torch import nn

from quack.gemm_interface import gemm, gemm_act


@dataclass(frozen=True)
class ExpertMLPShape:
    total_tokens: int
    num_experts: int
    in_features: int
    hidden_features: int
    out_features: int
    dtype: torch.dtype


def make_uniform_offsets(total_tokens: int, num_experts: int, device: str) -> torch.Tensor:
    counts = torch.full(
        (num_experts,), total_tokens // num_experts, device=device, dtype=torch.int32
    )
    counts[-1] += total_tokens - int(counts.sum().item())
    return counts.cumsum(0).to(torch.int32)


def cu_seqlens_from_offsets(offs: torch.Tensor) -> torch.Tensor:
    return torch.cat((offs.new_zeros(1), offs))


def expert_loop_mlp(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    outs = []
    start = 0
    for expert, end in enumerate(offs.tolist()):
        hidden = torch.relu(x[start:end] @ w1[expert])
        outs.append(hidden @ w2[expert])
        start = end
    return torch.cat(outs, dim=0)


def quack_expert_mlp_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    offs: torch.Tensor,
    *,
    tuned: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cu_seqlens_m = cu_seqlens_from_offsets(offs)
    preact, hidden = gemm_act(
        x,
        w1,
        activation="relu",
        cu_seqlens_m=cu_seqlens_m,
        store_preact=True,
        tuned=tuned,
    )
    return gemm(hidden, w2, cu_seqlens_m=cu_seqlens_m, tuned=tuned), hidden, preact


def fallback_dpreact(
    grad_out: torch.Tensor,
    w2: torch.Tensor,
    preact: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    return torch._grouped_mm(grad_out, w2.transpose(1, 2).contiguous(), offs) * (
        preact > 0
    ).to(grad_out.dtype)


def grouped_weight_grads(
    x: torch.Tensor,
    hidden: torch.Tensor,
    dpreact: torch.Tensor,
    grad_out: torch.Tensor,
    offs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dw1 = []
    dw2 = []
    start = 0
    for end in offs.tolist():
        dw1.append(x[start:end].t() @ dpreact[start:end])
        dw2.append(hidden[start:end].t() @ grad_out[start:end])
        start = end
    return torch.stack(dw1), torch.stack(dw2)


class QuackExpertReluMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        offs: torch.Tensor,
        tuned: bool,
    ) -> torch.Tensor:
        out, hidden, preact = quack_expert_mlp_forward(x, w1, w2, offs, tuned=tuned)
        ctx.save_for_backward(x, w1, w2, offs, hidden, preact)
        ctx.tuned = tuned
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w1, w2, offs, hidden, preact = ctx.saved_tensors
        dpreact = fallback_dpreact(grad_out, w2, preact, offs)

        dx = torch._grouped_mm(dpreact, w1.transpose(1, 2).contiguous(), offs)
        dw1, dw2 = grouped_weight_grads(x, hidden, dpreact, grad_out, offs)
        return dx, dw1, dw2, None, None


class QuackExpertReluMLP(nn.Module):
    def __init__(self, shape: ExpertMLPShape, *, device: str = "cuda") -> None:
        super().__init__()
        self.w1 = nn.Parameter(
            torch.empty(
                shape.num_experts,
                shape.in_features,
                shape.hidden_features,
                device=device,
                dtype=shape.dtype,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                shape.num_experts,
                shape.hidden_features,
                shape.out_features,
                device=device,
                dtype=shape.dtype,
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        offs: torch.Tensor,
        *,
        tuned: bool = False,
    ) -> torch.Tensor:
        return QuackExpertReluMLPFunction.apply(
            x, self.w1, self.w2, offs, tuned
        )


def make_case(shape: ExpertMLPShape):
    torch.manual_seed(0)
    device = "cuda"
    x = torch.randn(
        shape.total_tokens,
        shape.in_features,
        device=device,
        dtype=shape.dtype,
        requires_grad=True,
    )
    model = QuackExpertReluMLP(shape, device=device)
    offs = make_uniform_offsets(shape.total_tokens, shape.num_experts, device)
    grad_out = torch.randn(
        shape.total_tokens,
        shape.out_features,
        device=device,
        dtype=shape.dtype,
    )
    return x, model, offs, grad_out


def clone_for_reference(x: torch.Tensor, model: QuackExpertReluMLP):
    x_ref = x.detach().clone().requires_grad_()
    w1_ref = model.w1.detach().clone().requires_grad_()
    w2_ref = model.w2.detach().clone().requires_grad_()
    return x_ref, w1_ref, w2_ref


def check(shape: ExpertMLPShape, *, tuned: bool) -> None:
    x, model, offs, grad_out = make_case(shape)
    x_ref, w1_ref, w2_ref = clone_for_reference(x, model)

    actual = model(x, offs, tuned=tuned)
    expected = expert_loop_mlp(x_ref, w1_ref, w2_ref, offs)

    torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.05)
    actual.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=0.5, rtol=0.05)
    torch.testing.assert_close(model.w1.grad, w1_ref.grad, atol=0.5, rtol=0.05)
    torch.testing.assert_close(model.w2.grad, w2_ref.grad, atol=0.5, rtol=0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--hidden-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--tuned", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = ExpertMLPShape(
        total_tokens=args.tokens,
        num_experts=args.experts,
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        out_features=args.out_features,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
    )
    check(shape, tuned=args.tuned)
    print(
        "ok",
        f"tokens={shape.total_tokens}",
        f"experts={shape.num_experts}",
        f"in={shape.in_features}",
        f"hidden={shape.hidden_features}",
        f"out={shape.out_features}",
        f"dtype={shape.dtype}",
        f"tuned={args.tuned}",
    )


if __name__ == "__main__":
    main()

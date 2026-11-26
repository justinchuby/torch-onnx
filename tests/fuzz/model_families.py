"""Small randomized operator recipes inspired by modern model architectures.

These are independently implemented CPU subgraphs, not checkpoint or model
implementations. Parameters and recipe choices are serialized as ordinary nodes;
execution never regenerates weights or consults a model configuration.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

DOMAINS = ("decoder", "moe", "delta", "vision", "dit", "sparse_attention", "mhc")
ARITY = {
    "tensor_const": 0,
    "linear": 2,
    "linear_bias": 3,
    "rms_norm": 2,
    "layer_norm": 1,
    "silu": 1,
    "gelu": 1,
    "head_layout": 1,
    "merge_heads": 1,
    "rope": 3,
    "cached_gqa": 3,
    "softmax": 1,
    "topk_values": 1,
    "topk_indices": 1,
    "normalize_routes": 1,
    "expert_gather": 2,
    "expert_up": 2,
    "expert_down": 2,
    "weighted_scatter": 2,
    "causal_depthwise_conv": 2,
    "last_heads": 1,
    "l2_norm": 1,
    "decay": 1,
    "delta_update": 5,
    "state_read": 2,
    "patch_conv": 2,
    "resize_grid": 1,
    "patch_tokens": 1,
    "pool_tokens": 1,
    "adaln": 2,
    "gated_residual": 3,
    "local_scores": 2,
    "selected_values": 2,
    "attention_reduce": 2,
    "sinkhorn": 1,
    "hyper_mix": 2,
    "hyper_reduce": 2,
    "hyper_residual": 3,
}
OPS = frozenset(ARITY)


def apply(op, args, attrs):
    """Each recipe node is a fixed, pure composition of public Torch operations."""
    if op == "tensor_const":
        return torch.tensor(
            attrs["values"], dtype=getattr(torch, attrs["dtype"])
        ).reshape(attrs["shape"])
    a = args[0]
    if op == "linear":
        return F.linear(a, args[1].t())
    if op == "linear_bias":
        return F.linear(a, args[1].t(), args[2])
    if op == "rms_norm":
        return F.rms_norm(a, (a.shape[-1],), args[1], attrs["eps"])
    if op == "layer_norm":
        return F.layer_norm(a, (a.shape[-1],), eps=attrs["eps"])
    if op == "silu":
        return F.silu(a)
    if op == "gelu":
        return F.gelu(a, approximate=attrs["approximate"])
    if op == "head_layout":
        return a.reshape(a.shape[0], a.shape[1], attrs["heads"], -1).transpose(1, 2)
    if op == "merge_heads":
        return a.transpose(1, 2).reshape(a.shape[0], a.shape[2], -1)
    if op == "rope":
        angles = args[1].to(a.dtype).unsqueeze(-1) * args[2].unsqueeze(0)
        frequencies = torch.cat((angles, angles), dim=-1).unsqueeze(0).unsqueeze(0)
        left, right = a.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
        return a * frequencies.cos() + rotated * frequencies.sin()
    if op == "cached_gqa":
        keys, values = args[1:]
        repeat = attrs["repeat"]
        keys = keys.repeat_interleave(repeat, dim=1)
        values = values.repeat_interleave(repeat, dim=1)
        query_length, source_length = a.shape[2], keys.shape[2]
        positions = torch.arange(query_length) + source_length - query_length
        mask = torch.arange(source_length).unsqueeze(0) <= positions.unsqueeze(1)
        return F.scaled_dot_product_attention(a, keys, values, attn_mask=mask)
    if op == "softmax":
        return torch.softmax(a, dim=-1)
    if op in ("topk_values", "topk_indices"):
        result = torch.topk(a, attrs["k"], dim=-1, sorted=True)
        return result.values if op == "topk_values" else result.indices
    if op == "normalize_routes":
        return a / a.sum(dim=-1, keepdim=True)
    if op == "expert_gather":
        return a[args[1]]
    if op == "expert_up":
        return (a.unsqueeze(-2).unsqueeze(-2) @ args[1]).squeeze(-2)
    if op == "expert_down":
        return (a.unsqueeze(-2) @ args[1]).squeeze(-2)
    if op == "weighted_scatter":
        batch, length, selected, width = a.shape
        weighted = (a * args[1].unsqueeze(-1)).reshape(-1, width)
        tokens = torch.arange(batch * length).repeat_interleave(selected)
        index = tokens.unsqueeze(-1).expand(-1, width)
        output = a.new_zeros((batch * length, width)).scatter_add(0, index, weighted)
        return output.reshape(batch, length, width)
    if op == "causal_depthwise_conv":
        result = F.conv1d(a.transpose(1, 2), args[1], padding=2, groups=attrs["groups"])
        return result[:, :, : a.shape[1]].transpose(1, 2)
    if op == "last_heads":
        return a[:, -1, :].reshape(a.shape[0], attrs["heads"], -1)
    if op == "l2_norm":
        return F.normalize(a, dim=-1, eps=attrs["eps"])
    if op == "decay":
        return torch.exp(-F.softplus(a))
    if op == "delta_update":
        state, key, value, beta, decay = args
        state = state * decay
        prediction = (key.unsqueeze(-2) @ state).squeeze(-2)
        delta = (value - prediction) * torch.sigmoid(beta)
        return state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    if op == "state_read":
        return (a.unsqueeze(-2) @ args[1]).squeeze(-2)
    if op == "patch_conv":
        return F.conv2d(a, args[1], stride=attrs["stride"])
    if op == "resize_grid":
        return F.interpolate(
            a,
            size=(a.shape[-2] + 1, a.shape[-1] + 2),
            mode="bilinear",
            align_corners=False,
        )
    if op == "patch_tokens":
        return a.flatten(2).transpose(1, 2)
    if op == "pool_tokens":
        return a.mean(dim=1)
    if op == "adaln":
        shift, scale, _gate = args[1].chunk(3, dim=-1)
        return a * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    if op == "gated_residual":
        gate = args[2].chunk(3, dim=-1)[2]
        return a + args[1] * gate.unsqueeze(1)
    if op == "local_scores":
        query, keys = args
        scores = (query @ keys.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        positions = torch.arange(query.shape[2]) + keys.shape[2] - query.shape[2]
        distance = positions.unsqueeze(-1) - torch.arange(keys.shape[2]).unsqueeze(0)
        visible = (distance >= 0) & (distance < attrs["window"])
        return scores.masked_fill(~visible, float("-inf"))
    if op == "selected_values":
        indices = args[1]
        expanded = a.unsqueeze(2).expand(-1, -1, indices.shape[2], -1, -1)
        index = indices.unsqueeze(-1).expand(-1, -1, -1, -1, a.shape[-1])
        return torch.gather(expanded, 3, index)
    if op == "attention_reduce":
        return (a * args[1].unsqueeze(-1)).sum(dim=-2)
    if op == "sinkhorn":
        result = torch.softmax(a, dim=-1)
        for _ in range(attrs["iterations"]):
            result = result / (result.sum(dim=-2, keepdim=True) + attrs["eps"])
            result = result / (result.sum(dim=-1, keepdim=True) + attrs["eps"])
        return result
    if op == "hyper_mix":
        return torch.einsum("btsk,btkd->btsd", a, args[1])
    if op == "hyper_reduce":
        return (a * torch.sigmoid(args[1]).unsqueeze(-1)).sum(dim=-2)
    if op == "hyper_residual":
        return a + args[1].unsqueeze(-2) * torch.sigmoid(args[2]).unsqueeze(-1)
    raise ValueError(f"Unknown architecture operation {op!r}")


def validate_inputs(domain, specs):
    """Reject malformed replay input relationships before any export is attempted."""
    shapes = [spec["shape"] for spec in specs]
    if any(
        spec["dtype"] != "float32"
        for i, spec in enumerate(specs)
        if not (domain == "decoder" and i == 1)
    ):
        raise ValueError("Model recipes require float32 tensors")
    ranks = {
        "decoder": [3, 1, 4, 4],
        "moe": [3, 3],
        "delta": [3, 3, 3, 4, 3, 4],
        "vision": [4],
        "dit": [3, 2],
        "sparse_attention": [4, 4, 4],
        "mhc": [4, 4, 3],
    }[domain]
    if [len(shape) for shape in shapes] != ranks:
        raise ValueError("Invalid model recipe input ranks/arity")
    if domain == "decoder":
        (batch, length, width), (positions,), cache, other = shapes
        if (
            min(batch, length, width, cache[1], cache[3]) < 1
            or cache[0] != batch
            or cache != other
            or positions != length
            or width % (cache[1] * cache[3])
            or cache[3] % 2
            or specs[1]["dtype"] != "int64"
        ):
            raise ValueError("Invalid decoder/cache dimensions")
    elif any(min(shape) < 1 for shape in shapes):
        raise ValueError("Model recipe dimensions must be positive")
    elif domain == "moe":
        if shapes[0][:2] != shapes[1][:2] or shapes[1][-1] < 2:
            raise ValueError("Invalid MoE token/router dimensions")
    elif domain == "delta":
        (batch, _length, width), key, value, state, beta, decay = shapes
        if (
            key != value
            or key[0] != batch
            or width != key[1] * key[2]
            or state != [batch, key[1], key[2], key[2]]
            or beta != [batch, key[1], 1]
            or decay != [batch, key[1], 1, 1]
        ):
            raise ValueError("Invalid gated delta state dimensions")
    elif domain == "vision":
        if min(shapes[0][-2:]) < 4:
            raise ValueError("Patch recipe requires spatial dimensions >= 4")
    elif domain == "dit":
        if shapes[1] != [shapes[0][0], 3 * shapes[0][2]]:
            raise ValueError("Invalid adaptive normalization conditioning")
    elif domain == "sparse_attention":
        query, key, value = shapes
        if (
            query[:2] != key[:2]
            or query[-1] != key[-1]
            or key != value
            or key[2] < query[2] + 2
        ):
            raise ValueError(
                "Sparse recipe needs matching heads and enough visible keys"
            )
    elif domain == "mhc":
        batch, length, streams, _width = shapes[0]
        if shapes[1] != [batch, length, streams, streams] or shapes[2] != [
            batch,
            length,
            streams,
        ]:
            raise ValueError("Invalid hyper-connection dimensions")


class _Recipe:
    def __init__(self, seed, count):
        self.rng = random.Random(seed)
        self.count = count
        self.nodes = []

    def node(self, op, *args, **attrs):
        self.nodes.append(
            dict(
                op=op,
                args=list(args),
                attrs=attrs,
                dtype="int64" if op == "topk_indices" else "float32",
            )
        )
        return self.count + len(self.nodes) - 1

    def values(self, shape, *, kind="normal", dtype="float32", layout="contiguous"):
        size = math.prod(shape)
        values = [self.rng.uniform(-0.5, 0.5) for _ in range(size)]
        if kind == "gamma":
            values = [1 + value * 0.1 for value in values]
        elif kind == "weight":
            values = [value / math.sqrt(max(shape[-2], 1)) for value in values]
        return dict(shape=list(shape), dtype=dtype, values=values, layout=layout)

    def weight(self, shape, *, kind="weight"):
        spec = self.values(shape, kind=kind)
        return self.node(
            "tensor_const", shape=spec["shape"], values=spec["values"], dtype="float32"
        )


def generate(seed, *, domain, opset, dynamic, output_mode):
    # Local import keeps the case/runner modules usable as script entry points.
    if __package__:
        from .fuzzer import Case
    else:
        from fuzzer import Case

    counts = dict(decoder=4, moe=2, delta=6, vision=1, dit=2, sparse_attention=3, mhc=3)
    recipe = r = _Recipe(seed, counts[domain])
    width = r.rng.choice((8, 16))
    hidden = r.rng.choice((12, 24))
    heads, depth = 2, width // 2
    experts = r.rng.choice((3, 4, 6))
    selected = r.rng.choice((1, 2))
    eps = r.rng.choice((1e-5, 1e-6))
    streams = r.rng.choice((2, 3))
    layout = r.rng.choice(("contiguous", "transposed"))

    def inputs(batch, length, past):
        if domain == "decoder":
            result = [r.values((batch, length, width), layout=layout)]
            result.append(
                dict(
                    shape=[length],
                    dtype="int64",
                    layout="contiguous",
                    values=list(range(past, past + length)),
                )
            )
            result.extend(r.values((batch, 1, past, depth)) for _ in range(2))
            return result
        if domain == "moe":
            result = [
                r.values((batch, length, width), layout=layout),
                r.values((batch, length, experts)),
            ]
            scores = []
            for token in range(batch * length):
                order = list(range(experts))
                if seed % 3 == 1:
                    offset = token % experts
                    order = order[offset:] + order[:offset]
                elif seed % 3 == 2:
                    r.rng.shuffle(order)
                scores.extend(rank * 0.5 + r.rng.uniform(-0.01, 0.01) for rank in order)
            result[1]["values"] = scores
            return result
        if domain == "delta":
            shapes = [
                (batch, length, width),
                (batch, heads, depth),
                (batch, heads, depth),
                (batch, heads, depth, depth),
                (batch, heads, 1),
                (batch, heads, 1, 1),
            ]
        elif domain == "vision":
            shapes = [(batch, 3, 2 * length + 2, 2 * length + 4)]
        elif domain == "dit":
            shapes = [(batch, length, width), (batch, 3 * width)]
        elif domain == "sparse_attention":
            shapes = [
                (batch, heads, length, depth),
                (batch, heads, length + 2, depth),
                (batch, heads, length + 2, depth),
            ]
        else:
            shapes = [
                (batch, length, streams, width),
                (batch, length, streams, streams),
                (batch, length, streams),
            ]
        return [
            r.values(shape, layout=layout if i == 0 else "contiguous")
            for i, shape in enumerate(shapes)
        ]

    if domain == "decoder":
        gamma = r.weight((width,), kind="gamma")
        normalized = r.node("rms_norm", 0, gamma, eps=eps)
        query = r.node("linear", normalized, r.weight((width, width)))
        key = r.node("linear", normalized, r.weight((width, depth)))
        value = r.node("linear", normalized, r.weight((width, depth)))
        query = r.node("head_layout", query, heads=heads)
        key = r.node("head_layout", key, heads=1)
        value = r.node("head_layout", value, heads=1)
        frequencies = r.node(
            "tensor_const",
            shape=[depth // 2],
            values=[10000 ** (-2 * i / depth) for i in range(depth // 2)],
            dtype="float32",
        )
        query = r.node("rope", query, 1, frequencies)
        key = r.node("rope", key, 1, frequencies)
        key = r.node("cat", 2, key, dim=2)
        value = r.node("cat", 3, value, dim=2)
        attended = r.node("cached_gqa", query, key, value, repeat=heads)
        merged = r.node("merge_heads", attended)
        projected = r.node("linear", merged, r.weight((width, width)))
        residual = r.node("add", 0, projected)
        normalized = r.node("rms_norm", residual, gamma, eps=eps)
        gate = r.node("silu", r.node("linear", normalized, r.weight((width, hidden))))
        up = r.node("linear", normalized, r.weight((width, hidden)))
        hidden_value = r.node("mul", gate, up)
        down = r.node("linear", hidden_value, r.weight((hidden, width)))
        r.node("add", residual, down)
    elif domain == "moe":
        scores = r.node("softmax", 1)
        indices = r.node("topk_indices", scores, k=selected)
        weights = r.node("topk_values", scores, k=selected)
        weights = r.node("normalize_routes", weights)
        up = r.node("expert_gather", r.weight((experts, width, hidden)), indices)
        features = r.node("silu", r.node("expert_up", 0, up))
        down = r.node("expert_gather", r.weight((experts, hidden, width)), indices)
        features = r.node("expert_down", features, down)
        r.node("weighted_scatter", features, weights)
    elif domain == "delta":
        convolved = r.node(
            "causal_depthwise_conv", 0, r.weight((width, 1, 3)), groups=width
        )
        convolved = r.node("silu", convolved)
        query = r.node("l2_norm", r.node("last_heads", convolved, heads=heads), eps=eps)
        key = r.node("l2_norm", 1, eps=eps)
        decay = r.node("decay", 5)
        state = r.node("delta_update", 3, key, 2, 4, decay)
        r.node("state_read", query, state)
    elif domain == "vision":
        patches = r.node("patch_conv", 0, r.weight((width, 3, 2, 2)), stride=2)
        resized = r.node("resize_grid", patches)
        tokens = r.node("patch_tokens", resized)
        normalized = r.node("layer_norm", tokens, eps=eps)
        up = r.node(
            "linear_bias",
            normalized,
            r.weight((width, hidden)),
            r.weight((hidden,), kind="normal"),
        )
        up = r.node("gelu", up, approximate=r.rng.choice(("none", "tanh")))
        down = r.node(
            "linear_bias",
            up,
            r.weight((hidden, width)),
            r.weight((width,), kind="normal"),
        )
        r.node("pool_tokens", r.node("add", tokens, down))
    elif domain == "dit":
        normalized = r.node("layer_norm", 0, eps=eps)
        modulated = r.node("adaln", normalized, 1)
        up = r.node(
            "linear_bias",
            modulated,
            r.weight((width, hidden)),
            r.weight((hidden,), kind="normal"),
        )
        up = r.node("gelu", up, approximate=r.rng.choice(("none", "tanh")))
        down = r.node(
            "linear_bias",
            up,
            r.weight((hidden, width)),
            r.weight((width,), kind="normal"),
        )
        r.node("gated_residual", 0, down, 1)
    elif domain == "sparse_attention":
        scores = r.node("local_scores", 0, 1, window=r.rng.choice((2, 4)))
        indices = r.node("topk_indices", scores, k=selected)
        weights = r.node("softmax", r.node("topk_values", scores, k=selected))
        values = r.node("selected_values", 2, indices)
        r.node("attention_reduce", values, weights)
    else:
        transport = r.node("sinkhorn", 1, iterations=r.rng.choice((2, 4)), eps=eps)
        mixed = r.node("hyper_mix", transport, 0)
        combined = r.node("hyper_reduce", 0, 2)
        normalized = r.node(
            "rms_norm", combined, r.weight((width,), kind="gamma"), eps=eps
        )
        up = r.node("silu", r.node("linear", normalized, r.weight((width, hidden))))
        down = r.node("linear", up, r.weight((hidden, width)))
        r.node("hyper_residual", mixed, down, 2)

    case = Case(
        seed,
        inputs(2, 3, 2),
        recipe.nodes,
        version=3,
        domain=domain,
        output_mode=output_mode,
        opset=opset,
        input_probes=False,
        float32_tolerance=[1.3e-6, 1e-5],
    )
    if dynamic:
        names = (
            ["x", "y"]
            if len(case.inputs) == 2
            else [f"input_{i}" for i in range(len(case.inputs))]
        )
        axes = {names[i]: {"0": "batch"} for i in range(len(names))}
        axes[names[0]]["1" if domain != "vision" else "2"] = (
            "tokens" if domain != "vision" else "height"
        )
        if domain == "decoder":
            axes[names[1]] = {"0": "tokens"}
            axes[names[2]]["2"] = axes[names[3]]["2"] = "past"
        elif domain in ("moe", "mhc"):
            for name in names[1:]:
                axes[name]["1"] = "tokens"
        elif domain == "vision":
            axes[names[0]]["3"] = "width"
        elif domain == "sparse_attention":
            axes[names[0]] = {"0": "batch", "2": "query"}
            for name in names[1:]:
                axes[name]["2"] = "source"
        case.dynamic_shapes = axes
        symbols = {symbol for dims in axes.values() for symbol in dims.values()}
        case.dynamic_bounds = {
            symbol: [
                4 if symbol in ("height", "width") else 0 if symbol == "past" else 1,
                64,
            ]
            for symbol in symbols
        }
        case.additional_inputs = [inputs(3, 1, 0), inputs(1, 5, 3)]
    if domain != "sparse_attention":
        # Explicit finite edge profiles without manufacturing ambiguous TopK
        # ties in sparse attention. Values, not profile names, drive replay.
        scale = (0.0, 1e-4, 1.0, 10.0, 1.0)[seed % 5]
        for specs in [case.inputs, *case.additional_inputs]:
            specs[0]["values"] = [value * scale for value in specs[0]["values"]]
    return Case.from_json(case.to_json())

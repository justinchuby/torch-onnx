"""pytorch/pytorch#206939: runtime numerical support for ATen and prims specials."""

import onnx
import pytest
import torch

import torch_onnx

pytestmark = pytest.mark.compatibility

UNARY = [
    "erfinv",
    "lgamma",
    "digamma",
    "i0",
    "i1",
    "spherical_bessel_j0",
    "ndtri",
    "bessel_j0",
    "bessel_j1",
]


class Model(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def check(
    fn, args, opset=18, *, rtol=None, atol=None, backend="onnxruntime", expected=None
):
    expected = fn(*args) if expected is None else expected
    p = torch_onnx.export(Model(fn).eval(), args, opset_version=opset, verbose=False)
    try:
        onnx.checker.check_model(p.model_proto, full_check=True)
        (actual,) = p.call_reference(*args) if backend == "reference" else p(*args)
        double = expected.dtype == torch.float64
        torch.testing.assert_close(
            actual,
            expected,
            equal_nan=True,
            rtol=rtol if rtol is not None else (2e-11 if double else 2e-5),
            atol=atol if atol is not None else (2e-12 if double else 2e-6),
        )
        return actual
    finally:
        p.release()


@pytest.mark.parametrize("name", UNARY)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("opset", [18, 28])
def test_special_unary_domains_boundaries(name, dtype, opset):
    fn = (
        getattr(torch.special, name)
        if hasattr(torch.special, name)
        else getattr(torch, name)
    )
    if name == "erfinv":
        values = [
            -2.0,
            -1.0,
            -0.99999,
            -0.9,
            -0.5,
            -0.0,
            0.0,
            0.01,
            0.5,
            0.9,
            0.99999,
            1.0,
            2.0,
            float("nan"),
        ]
    elif name == "ndtri":
        values = [
            -1.0,
            0.0,
            1e-30,
            1e-10,
            0.001,
            0.1,
            0.1353352832366127,
            0.2,
            0.5,
            0.8,
            0.9,
            0.99999,
            1.0,
            2.0,
            float("nan"),
        ]
    else:
        values = [
            float("-inf"),
            -100.0,
            -10.0,
            -5.0,
            -2.5,
            -0.5,
            -0.0,
            0.0,
            1e-12,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
            8.0,
            10.0,
            50.0,
            100.0,
            float("inf"),
            float("nan"),
        ]
    check(
        fn,
        (torch.tensor(values, dtype=dtype),),
        opset,
        backend="reference"
        if name == "digamma" and dtype == torch.float64
        else "onnxruntime",
    )


@pytest.mark.parametrize("opset", [18, 28])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hypot_avoids_overflow_and_underflow(opset, dtype):
    small, large = (1e-30, 1e30) if dtype == torch.float32 else (1e-300, 1e300)
    x = torch.tensor([0.0, small, 3.0, large, float("inf")], dtype=dtype).reshape(-1, 1)
    y = torch.tensor([0.0, small, 4.0, large, float("nan")], dtype=dtype)
    check(torch.hypot, (x, y), opset)


@pytest.mark.parametrize("opset", [18, 28])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_nextafter_matches_adjacent_bits(opset, dtype):
    info = torch.finfo(dtype)
    x = torch.tensor(
        [
            -float("inf"),
            -info.max,
            -2.0,
            -1.0,
            -info.tiny,
            -info.tiny * info.eps,
            -0.0,
            0.0,
            info.tiny * info.eps,
            info.tiny,
            0.5,
            1.0,
            2.0,
            info.max,
            float("inf"),
        ],
        dtype=dtype,
    )
    for direction in [-float("inf"), 0.0, float("inf")]:
        y = torch.full_like(x, direction)
        expected = torch.nextafter(x, y)
        check(torch.nextafter, (x, y), opset, rtol=0, atol=0)
        # ORT folds Mul(x, 0) without preserving signed zero. Keep the spec-valid
        # graph and verify the exact zero sign with the ONNX reference backend.
        actual = check(
            torch.nextafter, (x, y), opset, rtol=0, atol=0, backend="reference"
        )
        integer = {
            torch.float16: torch.int16,
            torch.float32: torch.int32,
            torch.float64: torch.int64,
        }[dtype]
        torch.testing.assert_close(actual.view(integer), expected.view(integer))


@pytest.mark.parametrize("complement", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("opset", [18, 28])
def test_incomplete_gamma_domains_and_broadcast(complement, dtype, opset):
    fn = torch.special.gammaincc if complement else torch.special.gammainc
    a = torch.tensor(
        [0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 10000.0], dtype=dtype
    ).reshape(-1, 1)
    x = torch.tensor(
        [0.0, 1e-10, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 10000.0], dtype=dtype
    )
    # PyTorch CPU double at a=x=100 differs from the true CDF by 1.85e-10.
    check(
        fn,
        (a, x),
        opset,
        rtol=3e-5 if dtype == torch.float32 else 5e-10,
        backend="reference" if dtype == torch.float64 else "onnxruntime",
    )


@pytest.mark.parametrize("complement", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_incomplete_gamma_large_parameters_and_tails(complement, dtype):
    fn = torch.special.gammaincc if complement else torch.special.gammainc
    # Independent 80-digit quadrature of exp((a-1)*log(t)-t-lgamma(a)),
    # with t=a+sqrt(a)*z. PyTorch's 2000-term series is inaccurate in the
    # far lower tail at a=1e8, so that float result is not used as the oracle.
    a = torch.tensor([1e4, 1e6, 1e8], dtype=dtype).reshape(-1, 1)
    x = a + torch.sqrt(a) * torch.tensor([-8.0, -3.0, 0.0, 3.0, 8.0], dtype=dtype)
    reference_p = [
        [
            1.0130928844019716e-16,
            0.001234175584468492,
            0.5013298083399552,
            0.9985295051036143,
            0.9999999999999969,
        ],
        [
            5.24012281543083e-16,
            0.0013381041673135997,
            0.5001329807608725,
            0.9986382593537824,
            0.9999999999999992,
        ],
        [
            6.11569983219279e-16,
            0.0013487164491615506,
            0.5000132980760141,
            0.998648919898398,
            0.9999999999999993,
        ],
    ]
    reference_q = [
        [
            0.9999999999999999,
            0.9987658244155315,
            0.4986701916600448,
            0.0014704948963856813,
            3.1098199621688446e-15,
        ],
        [
            0.9999999999999994,
            0.9986618958326864,
            0.4998670192391274,
            0.0013617406462175915,
            7.370278579605256e-16,
        ],
        [
            0.9999999999999993,
            0.9986512835508384,
            0.49998670192398587,
            0.0013510801016019577,
            6.327903413122214e-16,
        ],
    ]
    expected = torch.tensor(reference_q if complement else reference_p, dtype=dtype)
    check(
        fn,
        (a.expand_as(x), x),
        rtol=3e-5 if dtype == torch.float32 else 3e-10,
        atol=1e-30,
        expected=expected,
        backend="reference" if dtype == torch.float64 else "onnxruntime",
    )


@pytest.mark.parametrize("complement", [False, True])
def test_incomplete_gamma_invalid_values_and_limits(complement):
    fn = torch.special.gammaincc if complement else torch.special.gammainc
    a = torch.tensor([-1.0, 0.0, 1.0, 2.0, float("inf"), float("nan")]).reshape(-1, 1)
    x = torch.tensor([-1.0, 0.0, 1.0, float("inf"), float("nan")])
    check(fn, (a, x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("opset", [18, 28])
def test_hurwitz_zeta_domains_broadcast_and_poles(dtype, opset):
    s = torch.tensor(
        [0.0, 0.5, 1.0, 1.1, 2.0, 3.0, 5.0, 10.0, 40.0], dtype=dtype
    ).reshape(-1, 1)
    q = torch.tensor(
        [-3.0, -2.5, -0.5, 0.0, 0.01, 0.1, 1.0, 2.0, 10.0, 100.0], dtype=dtype
    )
    check(torch.special.zeta, (s, q), opset)


@pytest.mark.parametrize(
    "name", ["lgamma", "digamma", "i0", "ndtri", "zeta", "gammainc", "gammaincc"]
)
def test_special_empty_inputs(name):
    fn = (
        getattr(torch.special, name)
        if hasattr(torch.special, name)
        else getattr(torch, name)
    )
    x = torch.empty(0, 2)
    args = (x, torch.ones(1, 2)) if name in ["zeta", "gammainc", "gammaincc"] else (x,)
    check(fn, args)


@pytest.mark.parametrize(
    "name",
    [
        "hypot",
        "erf_inv",
        "bessel_i0",
        "bessel_i1",
        "lgamma",
        "digamma",
        "spherical_bessel_j0",
        "ndtri",
        "bessel_j0",
        "bessel_j1",
        "igamma",
        "igammac",
        "nextafter",
        "zeta",
    ],
)
def test_already_decomposed_prims_registration(name):
    fn = getattr(torch.ops.prims, name).default
    x = torch.tensor([0.25, 0.5, 0.75])
    args = (
        (x, torch.tensor([1.0, 2.0, 3.0]))
        if name in ["hypot", "igamma", "igammac", "nextafter", "zeta"]
        else (x,)
    )
    if name == "zeta":
        args = (torch.tensor([2.0, 3.0, 4.0]), args[1])
    check(fn, args)


@pytest.mark.parametrize(
    "fn",
    [torch.hypot, torch.special.zeta, torch.special.gammainc, torch.special.gammaincc],
)
def test_special_dynamic_broadcast_and_serialization(fn, tmp_path):
    a = torch.full((3, 1), 2.5)
    x = torch.tensor([0.25, 1.0, 3.0, 10.0])
    program = torch_onnx.export(
        Model(fn).eval(),
        (a, x),
        verbose=False,
        dynamic_shapes=(
            (
                {0: torch.export.Dim("rows")},
                {0: torch.export.Dim("columns")},
            ),
        ),
    )
    try:
        path = tmp_path / "special.onnx"
        program.save(path)
        onnx.checker.check_model(onnx.load(path), full_check=True)
        import onnxruntime

        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        for rows, columns in [(2, 7), (5, 2)]:
            left = torch.full((rows, 1), 2.5)
            right = torch.linspace(0.25, 10, columns)
            feeds = {
                session.get_inputs()[0].name: left.numpy(),
                session.get_inputs()[1].name: right.numpy(),
            }
            actual = torch.from_numpy(session.run(None, feeds)[0])
            torch.testing.assert_close(actual, fn(left, right), rtol=3e-5, atol=2e-6)
    finally:
        program.release()


@pytest.mark.parametrize(
    "dtype,integer", [(torch.float32, torch.int32), (torch.float64, torch.int64)]
)
def test_nextafter_random_ieee_bit_patterns(dtype, integer):
    generator = torch.Generator().manual_seed(206939)
    info = torch.iinfo(integer)
    x = torch.randint(
        info.min, info.max, (512,), generator=generator, dtype=integer
    ).view(dtype)
    y = torch.where(torch.arange(512) % 2 == 0, torch.inf, -torch.inf).to(dtype)
    actual = check(torch.nextafter, (x, y), backend="reference", rtol=0, atol=0)
    finite = ~torch.isnan(x)
    torch.testing.assert_close(
        actual[finite].view(integer), torch.nextafter(x, y)[finite].view(integer)
    )

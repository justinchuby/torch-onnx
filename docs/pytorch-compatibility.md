# PyTorch compatibility

The package depends on `torch` without a version constraint. Installing the
exporter does not require replacing an existing PyTorch installation. Python
3.11 or newer is required independently of the selected PyTorch release.

The compatibility workflow exports models, checks their ONNX graphs, executes
them with ONNX Runtime, and compares against eager PyTorch. It uses public API
tests independently of PyTorch's internal test utilities and optional packages.
The full development suite continues to run on the current development baseline.

## Tested releases

The rolling window was refreshed on 2026-11-26. Every stable release published
since 2025-11-26 is included, including patch releases:

| PyTorch | Published |
| --- | --- |
| 2.10.0 | 2026-01-21 |
| 2.11.0 | 2026-03-23 |
| 2.12.0 | 2026-05-13 |
| 2.12.1 | 2026-06-18 |
| 2.13.0 | 2026-07-08 |
| 2.14.0 | 2026-09-02 |
| 2.14.1 | 2026-09-30 |
| 2.15.0 | 2026-10-28 |

Dates come from the [official PyTorch releases](https://github.com/pytorch/pytorch/releases).
The shared manifest `SUPPORTED_TORCH_VERSIONS` in
`src/torch_onnx/_torch_compat.py` drives CI. Update it and this table when refreshing
the annual window; the workflow does not query the network to change its matrix.

## Runtime behavior

Version differences alone do not reject export. Unvalidated release series or
missing capabilities produce one compatibility warning per process, naming the
tested releases and suggesting an upgrade. Patch releases in a validated minor
series do not trigger a warning solely for their patch number. Actual capture,
translation, or runtime failures still propagate normally.

Capability checks and release adapters live in `_torch_compat.py`. For example,
older releases lack the dynamic LSTM/GRU loop decomposition helpers. Ordinary
dynamic models can still export; dynamic recurrent models receive a warning
and use their installed PyTorch's capture behavior. No global PyTorch APIs are
monkeypatched to emulate newer releases.

Exporter version numbers follow the current development PyTorch major/minor,
with an independent patch increment for each exporter release. This release
number does not constrain which PyTorch versions can use the exporter.

## Selecting compatibility tests

Run `python -m pytest -m compatibility` to discover the cross-version suite.
Add `pytestmark = pytest.mark.compatibility` to a public-API test module to
include all its tests; function and class decorators work too. Use the explicit
`pytest.mark.compatibility` spelling so the pre-import collector can recognize
it. New marked modules are discovered without editing the workflow's file list.

Pytest normally imports every test module before applying `-m`. For the exact
`-m compatibility` selection, the root `conftest.py` scans syntax for this marker
before importing modules. This keeps baseline-only private PyTorch test APIs and
optional dependencies out of older environments. Marked modules still fail
normally on import errors, and syntax errors remain visible. Other marker
expressions use normal pytest collection.

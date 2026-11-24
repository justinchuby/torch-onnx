"""Public exceptions and warnings for the standalone exporter."""


class OnnxExporterWarning(UserWarning):
    """Warning during ONNX export."""


class OnnxExporterError(RuntimeError):
    """Error during export."""

    pass


class TorchExportError(OnnxExporterError):
    """Error during torch.export.export."""

    pass


class ConversionError(OnnxExporterError):
    """Error during ONNX conversion."""

    pass


class DispatchError(ConversionError):
    """Error during ONNX Function dispatching."""

    pass


class GraphConstructionError(ConversionError):
    """Error during graph construction."""

    pass

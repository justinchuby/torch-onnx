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
    """Error during ONNX Funtion dispatching."""

    pass


class GraphConstructionError(ConversionError):
    """Error during graph construction."""

    pass

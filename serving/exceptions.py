class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested before the model is ready."""


class InferenceError(RuntimeError):
    """Raised when model inference fails."""
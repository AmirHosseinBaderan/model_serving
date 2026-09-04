class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested before the model is ready."""
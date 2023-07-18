class ReplicateException(Exception):
    pass


class ModelError(ReplicateException):
    """An error from user's code in a model."""


class ReplicateError(ReplicateException):
    """An error from llm_replicate.vendored_replicate."""

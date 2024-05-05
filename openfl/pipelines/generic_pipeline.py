"""GenericPipeline module."""

from .pipeline import TransformationPipeline, Float32NumpyArrayToBytes


class GenericPipeline(TransformationPipeline):
    """The data pipeline without any compression."""

    def __init__(self, nn=True, **kwargs):
        """Initialize."""
        super(GenericPipeline, self).__init__(transformers=[Float32NumpyArrayToBytes(nn)], nn=nn,  **kwargs)
        self.nn = nn

    def is_nn(self):
        return self.nn
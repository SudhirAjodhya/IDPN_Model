from src.models.downsamplers.downsampler import Downsampler
import torch

class MaxPoolDownsampler(Downsampler):
    """
    Implements max pooling by overriding forward method of base downsampler class

    ?Args: 
        base downsampler class

    ?Return:
        Image downsampled by the given scale_factor
    """

    def __init__(self, scale_factor=2):
        super().__init__(scale_factor)
        self.pool = torch.nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        return self.pool(x)
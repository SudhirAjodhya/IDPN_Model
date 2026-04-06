import torch
from abc import abstractmethod

class Downsampler(torch.nn.Module):
    """
    Base abstract downsampler class for all downsampling modules

    Provides a consisten interface for the classifier

    ?Attributes:
        scale_factor (int): The factor by which the spatial dimensions (H, W) will be reduced. 
        A scale factor of 2 turns a 128 x 128 image into 64 x 64s

    """

    def __init__(self, scale_factor = 2):
        """
        Initializes base downsampler

        ?Args:
            scale_factor (int):  spatial resolution reduction factor - default is 2
        """

        super().__init__()  # Triggers PyTorch nn.Module initialization
        self.scale_factor = scale_factor

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: 

        """
        Defines the computation performed at every call.

        ?Args:
            x (torch.Tensor): Input feature map of shape (Batch, Channels, H, W).

        ?Returns:
            torch.Tensor: Downsampled feature map of shape - (Batch, New_channels, H/scale_factor, W/scale_factor) 

        *Note:
            This method is decorated with @abstractmethod, meaning this class cannot be instantiated directly. You MUST
            crete a chlid class that overrides this method.
 
        """
        pass

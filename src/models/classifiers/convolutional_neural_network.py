import torch
import torch.nn as nn

class ConvolutionalNeuralNetwork(nn.Module):
    """
    The base Convolutional Neural Network. Designed to accept any downsampling module
    """

    def __init__(self, downsampler_class, num_classes=1):
        super().__init__()
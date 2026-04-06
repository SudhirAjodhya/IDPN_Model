import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os


from src.models.downsamplers.static.max_pool import MaxPoolDownsampler

def run_visual_test(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        print("Make sure you have placed an image in your data_sets/raw/ folder.")
        return

    img = Image.open(image_path)#.convert('L') # convert image to grey scale : CXRs are single-channel
 
    # Convert to Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Add batch dimension: (Channel, H, W) -> (1, Channel, H, W)
    input_tensor = transform(img).unsqueeze(0)

    # Downsampler
    scale_factor = 10
    ds = MaxPoolDownsampler(scale_factor=scale_factor)

    with torch.no_grad():
        output_tensor = ds(input_tensor)

    print(f"Original Shape: {input_tensor.shape}")
    print(f"Downsampled Shape: {output_tensor.shape}")

    # Visualize Difference between original and Downsampled image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Remove batch/channel dims for plotting: .squeeze()
    axes[0].imshow(input_tensor.squeeze().numpy(), cmap='gray')
    axes[0].set_title(f"Original X-Ray ({input_tensor.shape[2]}x{input_tensor.shape[3]}))")
    axes[0].axis('off')

    axes[1].imshow(output_tensor.squeeze().numpy(), cmap='gray')
    axes[1].set_title(f"Max Pooled ({output_tensor.shape[-2]}x{output_tensor.shape[-1]})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    IMAGE_PATH = "data_sets/JSRT/images/images/JPCLN001.png" 
    run_visual_test(IMAGE_PATH)
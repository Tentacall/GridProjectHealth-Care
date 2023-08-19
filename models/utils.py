import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from math import pi
import os


def display(image_tensor, output_dir, file_name,  save = False, detach=False):
    # Assuming image_tensor is of shape [1, 3, 224, 224]
    assert len(image_tensor.shape) == 4, "Invalid input shape"
    
    if save and  not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if detach:
        image_np = image_tensor.detach().cpu().numpy()[0]  # Convert to NumPy array and remove singleton dimensions
    else:
        image_np = image_tensor.squeeze().detach().numpy()

    num_layers = image_np.shape[0]  # Number of layers (channels)
    num_rows = (num_layers + 2) // 3  # Calculate the number of rows needed

    # Create a grid of subplots for each layer
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    # Flatten axes if we have just one row
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    # Display each layer as a subplot
    for i in range(num_layers):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(image_np[i], cmap = "gray")
        axes[row, col].set_title(f"Layer {i + 1}")
        axes[row, col].axis('off')  # Turn off axis ticks and labels
            
            
    if save:
        img_filename = os.path.join(output_dir, f"{file_name}.png")
        plt.savefig(img_filename, bbox_inches='tight')
        plt.close(fig)
        return
    plt.tight_layout()
    plt.show()
    
def loading_bar( current_value, total_value, bar_length=40):
    progress = min(1.0, current_value / total_value)
    arrow = '■' * int(progress * bar_length)
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\r[{arrow}{spaces}] {int(progress * 100)}%', end='', flush=True)
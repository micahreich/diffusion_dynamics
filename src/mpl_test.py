import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
N_samples = 6  # Should be even
arr = np.random.rand(N_samples, 2, 128)  # Random data for testing

# Create a wide figure with 1 row and N_samples columns
fig, axes = plt.subplots(1, N_samples, figsize=(4 * N_samples, 4), sharey=True)

for i in range(N_samples):
    ax_top = axes[i].inset_axes([0, 0.55, 1, 0.4])  # Top subplot
    ax_bottom = axes[i].inset_axes([0, 0.1, 1, 0.4])  # Bottom subplot
    
    # Plot data
    ax_top.plot(arr[i, 0, :])
    ax_top.set_xticks([])  # Hide x-ticks on top subplot
    
    ax_bottom.plot(arr[i, 1, :])
    
    # Hide outer subplot borders
    axes[i].axis("off")
    axes[i].set_title(f"Sample {i + 1}")

# Adjust layout
plt.tight_layout()
plt.savefig("mpl_test.png")

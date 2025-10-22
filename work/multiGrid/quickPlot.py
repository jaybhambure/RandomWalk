#!/usr/bin/env python3
# Panel 1: Initial Gaussian on fine grid (tight, centered at ~50)
# Panel 2: Initial scaled Gaussian on coarse grid (tight, centered at ~25)
# Panel 3: Density histogram showing walker distribution
# Panel 4: Final coarse positions (spread out from diffusion)
# Panel 5: Final refined positions (spread out, back on fine grid)
# Panel 6: Before vs After comparison (shows diffusion effect)

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Read data from HDF5 file
with h5py.File('fine_grid_walkers.h5', 'r') as f:
    fine_walker_positions = f['fine_walker_positions'][:]
    coarse_walker_positions = f['coarse_walker_positions'][:]
    coarse_density = f['coarse_density'][:]
    coarse_walker_final = f['coarse_walker_final'][:]
    refined_walker_final = f['refined_walker_final'][:]

print(f"Loaded all datasets:")
print(f"  Initial fine positions: {len(fine_walker_positions)}")
print(f"  Initial coarse positions: {len(coarse_walker_positions)}")
print(f"  Final coarse positions: {len(coarse_walker_final)}")
print(f"  Final refined positions: {len(refined_walker_final)}")

# Create subplot layout (2x3 for more plots)
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Initial fine grid walker positions
ax1.hist(fine_walker_positions, bins=30, density=True, alpha=0.7, 
         color='skyblue', edgecolor='black')
ax1.set_title('1. Initial Fine Grid (0-100)')
ax1.set_xlabel('Position')
ax1.set_ylabel('Density')
ax1.grid(True, alpha=0.3)

# Plot 2: Initial coarse grid walker positions
ax2.hist(coarse_walker_positions, bins=15, density=True, alpha=0.7, 
         color='lightcoral', edgecolor='black')
ax2.set_title('2. Initial Coarse Grid (0-50)')
ax2.set_xlabel('Position')
ax2.set_ylabel('Density')
ax2.grid(True, alpha=0.3)

# Plot 3: Coarse grid density
x_coarse = np.arange(len(coarse_density))
ax3.bar(x_coarse, coarse_density, alpha=0.7, color='lightgreen', edgecolor='black')
ax3.set_title('3. Coarse Grid Density')
ax3.set_xlabel('Cell')
ax3.set_ylabel('Walker Count')
ax3.grid(True, alpha=0.3)

# Plot 4: Final coarse grid positions (after random walk)
ax4.hist(coarse_walker_final, bins=15, density=True, alpha=0.7, 
         color='orange', edgecolor='black')
ax4.set_title('4. Final Coarse Grid (after RW)')
ax4.set_xlabel('Position')
ax4.set_ylabel('Density')
ax4.grid(True, alpha=0.3)

# Plot 5: Final refined positions
ax5.hist(refined_walker_final, bins=30, density=True, alpha=0.7, 
         color='purple', edgecolor='black')
ax5.set_title('5. Final Refined Fine Grid')
ax5.set_xlabel('Position')
ax5.set_ylabel('Density')
ax5.grid(True, alpha=0.3)

# Plot 6: Before vs After comparison
ax6.hist(fine_walker_positions, bins=30, density=True, alpha=0.5, 
         color='skyblue', label='Initial Fine', edgecolor='blue')
ax6.hist(refined_walker_final, bins=30, density=True, alpha=0.5, 
         color='purple', label='Final Refined', edgecolor='purple')
ax6.set_title('6. Before vs After Comparison')
ax6.set_xlabel('Position')
ax6.set_ylabel('Density')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
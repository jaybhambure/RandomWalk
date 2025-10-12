import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load the initial and final positions from the HDF5 file
file_path = "walker_data.h5"
with h5py.File(file_path, "r") as h5file:
    initial_positions = h5file["initial_positions"][:]
    final_positions = h5file["final_positions"][:]

# Plot the initial and final positions
plt.figure(figsize=(12, 6))

# Plot initial positions
plt.hist(initial_positions, bins=100, alpha=0.5, color='green', edgecolor='black', label='Initial Positions', density=True)

# Plot final positions
plt.hist(final_positions, bins=100, alpha=0.7, color='blue', edgecolor='black', label='Final Positions', density=True)

plt.title("Initial and Final Positions of Random Walkers")
plt.xlabel("Position")
plt.ylabel("# of walkers (normalized)")
plt.legend()
plt.grid(True)
plt.show()
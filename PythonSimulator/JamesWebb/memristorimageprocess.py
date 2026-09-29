#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

# Link local physics engine
simulator_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if simulator_path not in sys.path:
    sys.path.append(simulator_path)

from molmem_lib import MolmemSimulator

import numpy as np
import cv2
import matplotlib.pyplot as plt

fig_dir = os.path.join(os.path.dirname(__file__), 'Figures')
os.makedirs(fig_dir, exist_ok=True)

# Optional: If you are running this in a Jupyter Notebook, this ensures the plot shows up inline
pass

def image_to_frequency_domain(image_path):
    # 1. Load the image using the provided path and convert to RGB
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found. Please check the file path.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Separate the color planes (Red, Green, Blue)
    r, g, b = cv2.split(image)

    # Define a helper function to perform the Fourier Transform
    def apply_fft(channel):
        # Apply the 2D Discrete Fourier Transform
        f_transform = np.fft.fft2(channel)
        
        # Shift the zero-frequency components to the center
        f_shift = np.fft.fftshift(f_transform)
        return f_shift

    # 3. Apply the FFT to each color plane
    freq_r = apply_fft(r)
    freq_g = apply_fft(g)
    freq_b = apply_fft(b)

    # 4. Calculate the Magnitude Spectrum for visualization
    def get_magnitude(freq_data):
        return 20 * np.log(np.abs(freq_data) + 1)

    mag_r = get_magnitude(freq_r)
    mag_g = get_magnitude(freq_g)
    mag_b = get_magnitude(freq_b)

    # --- Plotting the Results ---
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original color planes
    axs[0, 0].imshow(r, cmap='Reds')
    axs[0, 0].set_title('Red Plane (Spatial)')
    axs[0, 1].imshow(g, cmap='Greens')
    axs[0, 1].set_title('Green Plane (Spatial)')
    axs[0, 2].imshow(b, cmap='Blues')
    axs[0, 2].set_title('Blue Plane (Spatial)')

    # Frequency domain representation
    axs[1, 0].imshow(mag_r, cmap='gray')
    axs[1, 0].set_title('Red Plane (Frequency Magnitude)')
    axs[1, 1].imshow(mag_g, cmap='gray')
    axs[1, 1].set_title('Green Plane (Frequency Magnitude)')
    axs[1, 2].imshow(mag_b, cmap='gray')
    axs[1, 2].set_title('Blue Plane (Frequency Magnitude)')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure_1.png"), bbox_inches="tight"); plt.close()

    return freq_r, freq_g, freq_b

# --- Run the Code ---
# This line actually executes the function using your file path!
if __name__ == '__main__':
    print("\n[Step 1] Loading image and converting to frequency domain...")
    freq_r, freq_g, freq_b = image_to_frequency_domain(os.path.join(os.path.dirname(__file__), 'PillarsofCreation.png'))
    print("Fourier Transform analysis complete. Saved spatial and frequency magnitudes visualization to Figures/figure_1.png.")
    
    print(f"Red plane frequency dimensions: {freq_r.shape}")
    print(f"Green plane frequency dimensions: {freq_g.shape}")
    print(f"Blue plane frequency dimensions: {freq_b.shape}")


# In[3]:


import numpy as np

def prepare_crossbar_inputs(freq_2d_matrix):
    # 1. Flatten the 2D matrix into a 1D vector
    # This turns a grid of e.g., 1000x1000 into a single line of 1,000,000 numbers
    flat_vector = freq_2d_matrix.flatten()
    
    # 2. Check if the total length is perfectly divisible by 64
    total_elements = len(flat_vector)
    remainder = total_elements % 64
    
    # 3. If there is a remainder, pad the end with zeros
    if remainder != 0:
        padding_needed = 64 - remainder
        # Pad with complex zeros to match the FFT data type
        flat_vector = np.pad(flat_vector, (0, padding_needed), mode='constant', constant_values=0j)
        
    # 4. Calculate how many 64-element steps we now have
    num_steps = len(flat_vector) // 64
    
    # 5. Reshape into an array of column matrices
    # The shape will be (number_of_steps, 64, 1)
    crossbar_inputs = flat_vector.reshape(num_steps, 64, 1)
    
    return crossbar_inputs

# --- Apply the function to your frequency data ---
# Assuming you already have freq_r, freq_g, and freq_b from the previous step:
if __name__ == '__main__':
    print("\n[Step 2] Formatting frequency domain coefficients into 64-element crossbar input vectors...")
    crossbar_inputs_r = prepare_crossbar_inputs(freq_r)
    crossbar_inputs_g = prepare_crossbar_inputs(freq_g)
    crossbar_inputs_b = prepare_crossbar_inputs(freq_b)
    print("Crossbar input formatting complete.")
    
    # --- Verify the results ---
    print(f"Total steps required for Red plane: {crossbar_inputs_r.shape[0]}")
    print(f"Shape of a single input step: {crossbar_inputs_r[0].shape}")
    
    print(f"Total steps required for Green plane: {crossbar_inputs_g.shape[0]}")
    print(f"Shape of a single input step: {crossbar_inputs_g[0].shape}")
    
    print(f"Total steps required for Blue plane: {crossbar_inputs_b.shape[0]}")
    print(f"Shape of a single input step: {crossbar_inputs_b[0].shape}")


# In[5]:


# 1. Extract the Phase Spectrum
# np.angle returns the angle (phase) of the complex numbers
if __name__ == '__main__':
    print("\n[Step 3] Extracting phase spectrum and generating visualization...")
    phase_r = np.angle(freq_r)
    phase_g = np.angle(freq_g)
    phase_b = np.angle(freq_b)
    
    # 2. Plotting just the Phase
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(phase_r, cmap='gray')
    axs[0].set_title('Red Plane (Phase)')
    
    axs[1].imshow(phase_g, cmap='gray')
    axs[1].set_title('Green Plane (Phase)')
    
    axs[2].imshow(phase_b, cmap='gray')
    axs[2].set_title('Blue Plane (Phase)')
    
    for ax in axs:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure_2.png"), bbox_inches="tight"); plt.close()
    print("Phase spectrum visualization saved to Figures/figure_2.png.")


# In[6]:


import numpy as np
import matplotlib.pyplot as plt

def create_crossbar_conductance_matrix(Gmax, Gmin, N=64):
    """
    Creates a 64x64 conductance matrix for IDFT mapping based on the memristor's Gmax and Gmin.
    """
    # 1. Create the grid for our N x N matrix (p = rows, q = columns)
    p = np.arange(N).reshape(N, 1)
    q = np.arange(N).reshape(1, N)
    
    # 2. Calculate the IDFT twiddle factors
    # For Inverse DFT, W = e^(j * 2 * pi * p * q / N)
    W_idft = np.exp(1j * 2 * np.pi * p * q / N)
    
    # Extract the Real and Imaginary components (these range from -1 to 1)
    W_real = np.real(W_idft)
    W_imag = np.imag(W_idft)
    
    # 3. Map to the Crossbar Structure
    # Initialize an empty 64x64 matrix
    W_mapped = np.zeros((N, N))
    
    # Left Half: First N/2 columns get the Real part of the IDFT
    W_mapped[:, :N//2] = W_real[:, :N//2]
    
    # Right Half: Last N/2 columns get the Imaginary part of the IDFT
    # (Mapping the first N/2 columns of the imaginary data to the right side of the board)
    W_mapped[:, N//2:] = W_imag[:, :N//2]
    
    # 4. Scale to Physical Conductance
    # Map the [-1, 1] range to the [Gmin, Gmax] range
    G_mid = (Gmax + Gmin) / 2
    G_amp = (Gmax - Gmin) / 2
    
    G_matrix = (G_amp * W_mapped) + G_mid
    
    return G_matrix

# --- Execute the Code ---
# Using the values from the paper: Gmax is ~5.9 mS, Gmin is ~200 nS
if __name__ == '__main__':
    print("\n[Step 4] Creating 64x64 IDFT target conductance matrix...")
    Gmax_val = 5.9e-3  # 5.9 milli-Siemens
    Gmin_val = 200e-9  # 200 nano-Siemens
    
    # Generate the 64x64 crossbar state
    crossbar_conductance = create_crossbar_conductance_matrix(Gmax_val, Gmin_val, N=64)
    
    # --- Verification and Visualization ---
    print(f"Matrix Shape: {crossbar_conductance.shape}")
    print(f"Maximum Conductance: {np.max(crossbar_conductance):.2e} S")
    print(f"Minimum Conductance: {np.min(crossbar_conductance):.2e} S")
    
    # --- Verification and Visualization ---
    plt.figure(figsize=(8, 6))
    
    # ADDED pad=25 to push the main title up and out of the way
    plt.title("64x64 Memristor Crossbar Conductance Map", pad=25) 
    
    plt.imshow(crossbar_conductance, cmap='viridis')
    plt.colorbar(label='Conductance (Siemens)')
    plt.xlabel("Columns (Bit Lines)")
    plt.ylabel("Rows (Word Lines)")
    plt.axvline(x=31.5, color='white', linestyle='--', linewidth=2)
    
    # Adjusted coordinates slightly and added ha='center' for perfect alignment
    plt.text(15.5, -3, "Real Part", color='black', fontweight='bold', ha='center')
    plt.text(47.5, -3, "Imaginary Part", color='black', fontweight='bold', ha='center')
    
    plt.tight_layout() # Helps ensure nothing gets cut off at the edges
    plt.savefig(os.path.join(fig_dir, "figure_3.png"), bbox_inches="tight"); plt.close()
    print("Crossbar conductance mapping complete. Saved conductance visualization to Figures/figure_3.png.")


# In[8]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. HARDWARE SETUP
# ==========================================
def create_crossbar(N=64, Gmax=5.9e-3, Gmin=200e-9):
    p = np.arange(N).reshape(N, 1)
    q = np.arange(N).reshape(1, N)
    W_idft = np.exp(1j * 2 * np.pi * p * q / N)
    
    W_mapped = np.zeros((N, N))
    W_mapped[:, :N//2] = np.real(W_idft[:, :N//2])
    W_mapped[:, N//2:] = np.imag(W_idft[:, :N//2])
    
    G_mid = (Gmax + Gmin) / 2
    G_amp = (Gmax - Gmin) / 2
    G_matrix = (G_amp * W_mapped) + G_mid
    return G_matrix, G_mid, G_amp

# ==========================================
# 2. THE SIMULATOR (1D ROW-BY-ROW)
# ==========================================
def simulate_crossbar_batch(X_batch, G_matrix, G_mid, G_amp, N=64):
    V_re, V_im = np.real(X_batch), np.imag(X_batch)
    
    # Analog Matrix Multiplication
    I_1 = np.dot(V_re, G_matrix)
    I_2 = np.dot(V_im, G_matrix)
    
    # Baseline subtraction and rescaling
    M_1 = (I_1 - (np.sum(V_re) * G_mid)) / (G_amp * N)
    M_2 = (I_2 - (np.sum(V_im) * G_mid)) / (G_amp * N)
    
    I_re_L, I_re_R = M_1[:N//2], M_1[N//2:]
    I_im_L, I_im_R = M_2[:N//2], M_2[N//2:]
    
    # Symmetry Stitching
    pixels_left = I_re_L - I_im_R
    pixels_right = (I_re_L[1:N//2] + I_im_R[1:N//2])[::-1]
    
    W_nyquist = (-1)**np.arange(N)
    pixel_mid = np.dot(V_re, W_nyquist) / N
    
    return np.concatenate((pixels_left, [pixel_mid], pixels_right))

def process_block(block, G_mat, G_mid, G_amp, N=64):
    """Processes a single 64x64 color channel block through the hardware."""
    freq_block = np.zeros_like(block, dtype=complex)
    for i in range(N):
        freq_block[i, :] = np.fft.fft(block[i, :])
        
    flat_freq = freq_block.flatten().reshape(N, N)
    recon_flat = np.zeros(N * N)
    
    for i in range(N):
        recon_flat[i*N : (i+1)*N] = simulate_crossbar_batch(flat_freq[i], G_mat, G_mid, G_amp, N)
        
    return np.abs(recon_flat.reshape(N, N))

# ==========================================
# 3. GLOBAL FREQUENCY ENCODER (For Dashboard Only)
# ==========================================
def get_global_plot_data(channel):
    """Calculates global 1D FFTs just so we can visualize them in the plot."""
    freq = np.zeros_like(channel, dtype=complex)
    for i in range(channel.shape[0]):
        freq[i, :] = np.fft.fft(channel[i, :])
    
    freq_shifted = np.fft.fftshift(freq, axes=1)
    mag = np.log1p(np.abs(freq_shifted))
    phase = np.angle(freq_shifted)
    return mag, phase

# ==========================================
# 4. EXECUTION SCRIPT
# ==========================================
if __name__ == '__main__':
    print("\n[Step 5] Running mathematical mock hardware simulator on original image...")
    N = 64
    G_mat, G_mid, G_amp = create_crossbar(N=N, Gmax=5.9e-3, Gmin=200e-9)
    
    # Load the High-Res image
    image_path = os.path.join(os.path.dirname(__file__), 'PillarsofCreation.png')
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found! Check the path.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    orig_h, orig_w = image.shape[:2]
    print(f"Original Image Shape: {orig_h}x{orig_w}")
    
    r_glob, g_glob, b_glob = cv2.split(image) 
    mag_r, phase_r = get_global_plot_data(r_glob)
    mag_g, phase_g = get_global_plot_data(g_glob)
    mag_b, phase_b = get_global_plot_data(b_glob)
    
    # -- B. Hardware Prep: Padding for Tiling --
    pad_h = (N - (orig_h % N)) % N
    pad_w = (N - (orig_w % N)) % N
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    new_h, new_w = padded_img.shape[:2]
    reconstructed_padded = np.zeros_like(padded_img, dtype=np.float32)
    
    # -- C. Hardware Simulation Engine --
    print(f"Processing hardware simulation in {N}x{N} blocks... (This may take a moment)")
    total_blocks = (new_h // N) * (new_w // N)
    block_count = 0
    
    for y in range(0, new_h, N):
        for x in range(0, new_w, N):
            block_count += 1
            if block_count % 100 == 0:
                print(f"   Processed block {block_count} of {total_blocks}...")
                
            block = padded_img[y:y+N, x:x+N]
            r_blk, g_blk, b_blk = cv2.split(block)
            
            recon_r = process_block(r_blk, G_mat, G_mid, G_amp, N)
            recon_g = process_block(g_blk, G_mat, G_mid, G_amp, N)
            recon_b = process_block(b_blk, G_mat, G_mid, G_amp, N)
            
            reconstructed_padded[y:y+N, x:x+N] = cv2.merge([recon_r, recon_g, recon_b])
            
    # Crop back to original dimensions
    final_reconstruction = reconstructed_padded[:orig_h, :orig_w]
    
    # Normalize for display
    def normalize_step5(img):
        img_norm = img - np.min(img)
        return ((img_norm / np.max(img_norm)) * 255).astype(np.uint8)
        
    final_rgb = normalize_step5(final_reconstruction)
    print("Hardware simulation complete! Rendering dashboard...")
    
    # ==========================================
    # 5. THE 12-PANEL DASHBOARD
    # ==========================================
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"High-Res Hardware Reconstruction ({orig_h}x{orig_w})", fontsize=22, fontweight='bold')
    
    # Row 1: Original and Final
    ax_orig = plt.subplot2grid((4, 3), (0, 0))
    ax_orig.imshow(image)
    ax_orig.set_title("1. Original Image")
    ax_orig.axis('off')
    
    ax_final = plt.subplot2grid((4, 3), (0, 2))
    ax_final.imshow(final_rgb)
    ax_final.set_title("5. Hardware Reconstructed (Tiled)")
    ax_final.axis('off')
    
    # Row 2: Color Panes
    ax_r = plt.subplot2grid((4, 3), (1, 0))
    ax_r.imshow(r_glob, cmap='Reds')
    ax_r.set_title("2a. Red Spatial Pane")
    ax_r.axis('off')
    
    ax_g = plt.subplot2grid((4, 3), (1, 1))
    ax_g.imshow(g_glob, cmap='Greens')
    ax_g.set_title("2b. Green Spatial Pane")
    ax_g.axis('off')
    
    ax_b = plt.subplot2grid((4, 3), (1, 2))
    ax_b.imshow(b_glob, cmap='Blues')
    ax_b.set_title("2c. Blue Spatial Pane")
    ax_b.axis('off')
    
    # Row 3: Frequency Magnitude
    ax_mr = plt.subplot2grid((4, 3), (2, 0))
    ax_mr.imshow(mag_r, cmap='magma')
    ax_mr.set_title("3a. Red 1D Freq Magnitude")
    ax_mr.axis('off')
    
    ax_mg = plt.subplot2grid((4, 3), (2, 1))
    ax_mg.imshow(mag_g, cmap='magma')
    ax_mg.set_title("3b. Green 1D Freq Magnitude")
    ax_mg.axis('off')
    
    ax_mb = plt.subplot2grid((4, 3), (2, 2))
    ax_mb.imshow(mag_b, cmap='magma')
    ax_mb.set_title("3c. Blue 1D Freq Magnitude")
    ax_mb.axis('off')
    
    # Row 4: Frequency Phase
    ax_pr = plt.subplot2grid((4, 3), (3, 0))
    ax_pr.imshow(phase_r, cmap='hsv')
    ax_pr.set_title("4a. Red 1D Freq Phase")
    ax_pr.axis('off')
    
    ax_pg = plt.subplot2grid((4, 3), (3, 1))
    ax_pg.imshow(phase_g, cmap='hsv')
    ax_pg.set_title("4b. Green 1D Freq Phase")
    ax_pg.axis('off')
    
    ax_pb = plt.subplot2grid((4, 3), (3, 2))
    ax_pb.imshow(phase_b, cmap='hsv')
    ax_pb.set_title("4c. Blue 1D Freq Phase")
    ax_pb.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(fig_dir, "figure_4.png"), bbox_inches="tight"); plt.close()
    print("Saved 12-panel mathematical reconstruction dashboard to Figures/figure_4.png.")
    
    plt.imshow(G_mat, cmap='viridis')
    plt.colorbar(label='Conductance (Siemens)')
    plt.savefig(os.path.join(fig_dir, "figure_5.png"), bbox_inches="tight"); plt.close()
    print("Saved crossbar twiddle factor conductance matrix map to Figures/figure_5.png.")


# In[9]:


import sys
import os

# ==========================================
# 0. LINK LOCAL PHYSICS ENGINE
# ==========================================
# Tell Python exactly where the molmem_lib folder is located
simulator_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if simulator_path not in sys.path:
    sys.path.append(simulator_path)

# Now we can safely import your team's API alongside the standard libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from molmem_lib import MolmemSimulator

# ... (The rest of the IDFT hardware math and execution code continues here) ...


# In[ ]:


from molmem_lib import MolmemSimulator
import numpy as np

if __name__ == '__main__':
    print("\n[Step 6] Initializing MolMem Transient Physics Engine and calibrating limits...")
    # 1. Instantiate the Hardware Core
    sim = MolmemSimulator(instanceName="ImageIDFT_Core")
    
    # 2. Build the Netlist and Thread Allocator
    sim.addCrossbarMatrix(
        rows=64, 
        cols=64, 
        device_type="default",
        weightBits=8, 
        inputBits=8, 
        outputBits=8,
        multiThread=True, 
        detailedPrint=True,
        max_vWrite=5.0,
        max_pw=320e-9,
        architecture='1T1R',
        UpdatesPer="Column"
    )
    
    print("\n--- Running Physical Conductance Limit Calibration ---")
    
    # 3. Test Absolute Gmax (Program all devices to 1.0)
    print("Pulsing hardware to Maximum Potentiation...")
    sim.updateCrossbarWeights(np.ones((64, 64)), detailedPrint=False, fineTune=True)
    gMax_matrix, _ = sim.readCrossbarMatrix(detailedPrint=False)
    absolute_Gmax = np.max(gMax_matrix)
    
    # 4. Test Absolute Gmin (Program all devices to 0.0)
    print("Pulsing hardware to Maximum Depression...")
    sim.updateCrossbarWeights(np.zeros((64, 64)), detailedPrint=False, fineTune=True)
    gMin_matrix, _ = sim.readCrossbarMatrix(detailedPrint=False)
    absolute_Gmin = np.min(gMin_matrix)
    
    print(f"\nHardware Calibration Complete:")
    print(f"-> Absolute Physical Gmax: {absolute_Gmax:.2e} Siemens")
    print(f"-> Absolute Physical Gmin: {absolute_Gmin:.2e} Siemens")


# In[16]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from molmem_lib import MolmemSimulator # Your team's API

# ==========================================
# 1. HARDWARE MATH PREPARATION
# ==========================================
def create_target_weights(N=64):
    """Calculates IDFT and maps it strictly to [0.0, 1.0] for the molmem API."""
    p = np.arange(N).reshape(N, 1)
    q = np.arange(N).reshape(1, N)
    W_idft = np.exp(1j * 2 * np.pi * p * q / N)
    
    W_mapped = np.zeros((N, N))
    W_mapped[:, :N//2] = np.real(W_idft[:, :N//2])
    W_mapped[:, N//2:] = np.imag(W_idft[:, :N//2])
    
    # Map from [-1, 1] mathematical range to [0, 1] hardware programming range
    W_target = (W_mapped + 1.0) / 2.0
    return W_target

# ==========================================
# 2. MOLMEM BATCH SIMULATOR
# ==========================================
def simulate_crossbar_batch_api(X_batch, sim_instance, N=64):
    """Encodes arbitrary frequency batches to [0,1], runs inference, and reconstructs."""
    is_1d = (X_batch.ndim == 1)
    if is_1d:
        X_batch_2d = X_batch.reshape(1, -1)
    else:
        X_batch_2d = X_batch
        
    V_re = np.real(X_batch_2d)
    V_im = np.imag(X_batch_2d)
    
    def pass_bipolar(V):
        """Splits vectors into positive/negative, normalizes, and passes to hardware."""
        v_max = np.max(np.abs(V), axis=1, keepdims=True)
        v_max_div = np.where(v_max == 0, 1.0, v_max)
        
        # Split and normalize to strictly [0.0, 1.0] as required by passInput
        V_pos = np.clip(V, 0, None) / v_max_div
        V_neg = np.clip(-V, 0, None) / v_max_div
        
        # Execute the Transient Physics Forward Pass (PWM & ADC Integration)
        Y_pos = sim_instance.passInput(V_pos, detailedPrint=False)
        Y_neg = sim_instance.passInput(V_neg, detailedPrint=False)
        
        # Undo the [0, 1] weight mapping (Math: Y_true = 2 * Y_hw - sum(X))
        Y_math_pos = 2 * Y_pos - np.sum(V_pos, axis=1, keepdims=True)
        Y_math_neg = 2 * Y_neg - np.sum(V_neg, axis=1, keepdims=True)
        
        # Recombine bipolar outputs and undo the voltage scaling
        return (Y_math_pos - Y_math_neg) * v_max

    # Run Real and Imaginary batches
    M_1 = pass_bipolar(V_re) / N
    M_2 = pass_bipolar(V_im) / N
    
    # Symmetry Stitching (Remains identical to previous math)
    I_re_L, I_re_R = M_1[:, :N//2], M_1[:, N//2:]
    I_im_L, I_im_R = M_2[:, :N//2], M_2[:, N//2:]
    
    pixels_left = I_re_L - I_im_R
    pixels_right = (I_re_L[:, 1:N//2] + I_im_R[:, 1:N//2])[:, ::-1]
    
    W_nyquist = (-1)**np.arange(N)
    pixel_mid = (np.dot(V_re, W_nyquist) / N).reshape(-1, 1)
    
    recon = np.hstack((pixels_left, pixel_mid, pixels_right))
    if is_1d:
        return recon[0]
    return recon

def process_block_api(block, sim_instance, N=64):
    """Processes a 64x64 block through the molmem simulator."""
    freq_block = np.zeros_like(block, dtype=complex)
    for i in range(N):
        freq_block[i, :] = np.fft.fft(block[i, :])
        
    flat_freq = freq_block.flatten().reshape(N, N)
    recon_flat = simulate_crossbar_batch_api(flat_freq, sim_instance, N)
    
    return np.abs(recon_flat)

# ==========================================
# 3. EXECUTION ENGINE
# ==========================================
if __name__ == '__main__':
    print("\n[Step 7] Running block-by-block transient simulation using MolMem Physics Engine...")
    N = 64
    
    # A. Initialize molmem_lib Hardware
    print("Booting MolMem Transient Physics Engine...")
    mol_sim = MolmemSimulator(instanceName="ImageIDFT_Core")
    mol_sim.addCrossbarMatrix(rows=N, cols=N, device_type="default", weightBits=14, inputBits=14, outputBits=14, multiThread=True, detailedPrint=False, max_vWrite=5.0, max_pw=320e-9, architecture='1T1R', UpdatesPer="Device")
    
    # B. Program the Fourier Matrix
    print("Programming Crossbar with IDFT Conductance Matrix...")
    W_target = create_target_weights(N)
    # Using fineTune=False for speed during this massive image reconstruction
    mol_sim.updateCrossbarWeights(W_target, detailedPrint=True, fineTune=True)
    mol_sim.updateCrossbarWeights(W_target, detailedPrint=True, fineTune=True)
    
    # C. Load and Pad Image
    image_path = os.path.join(os.path.dirname(__file__), 'PillarsofCreation.png')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # ==========================================
    # TEST MODE: Downscale the massive 48MP image
    # ==========================================
    image = cv2.resize(image, (1000, 1000)) 
    
    orig_h, orig_w = image.shape[:2]
    pad_h = (N - (orig_h % N)) % N
    pad_w = (N - (orig_w % N)) % N
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    new_h, new_w = padded_img.shape[:2]
    reconstructed_padded = np.zeros_like(padded_img, dtype=np.float32)
    
    # D. Run the Image through the Physics Engine
    total_blocks = (new_h // N) * (new_w // N)
    block_count = 0
    
    print(f"Beginning Transient Inference on {total_blocks} blocks...")
    for y in range(0, new_h, N):
        for x in range(0, new_w, N):
            block_count += 1
            if block_count % 1 == 0:
                print(f"   Inference running on block {block_count} of {total_blocks}...")
                
            block = padded_img[y:y+N, x:x+N]
            r_blk, g_blk, b_blk = cv2.split(block)
            
            # Sending arrays directly into the differential equation solver!
            recon_r = process_block_api(r_blk, mol_sim, N)
            recon_g = process_block_api(g_blk, mol_sim, N)
            recon_b = process_block_api(b_blk, mol_sim, N)
            
            reconstructed_padded[y:y+N, x:x+N] = cv2.merge([recon_r, recon_g, recon_b])
            
    # E. Finalize and Display
    final_reconstruction = reconstructed_padded[:orig_h, :orig_w]
    
    def normalize_step7(img):
        img_norm = img - np.min(img)
        return ((img_norm / np.max(img_norm)) * 255).astype(np.uint8)
        
    final_rgb = normalize_step7(final_reconstruction)
    print("Transient simulation complete. Generating reconstruction comparison plot...")
    
    # Plotting the final output
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original High-Res Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(final_rgb)
    plt.title("MolMem Physically Reconstructed Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure_6.png"), bbox_inches="tight"); plt.close()
    print("Saved reconstructed comparison plot to Figures/figure_6.png.")


# In[17]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # ==========================================
    # 1. LOAD TRUE HIGH-RES ORIGINAL
    # ==========================================
    # We load this fresh because the 'image' variable in memory was downscaled
    true_orig = cv2.imread(image_path)
    true_orig = cv2.cvtColor(true_orig, cv2.COLOR_BGR2RGB)
    
    # ==========================================
    # 2. FAST GLOBAL FFT (Visuals Only)
    # ==========================================
    # This uses standard math on the downscaled 'image', executing instantly
    r_glob, g_glob, b_glob = cv2.split(image) 
    
    def get_plot_data(channel):
        freq = np.zeros_like(channel, dtype=complex)
        for i in range(channel.shape[0]):
            freq[i, :] = np.fft.fft(channel[i, :])
        freq_shifted = np.fft.fftshift(freq, axes=1)
        mag = np.log1p(np.abs(freq_shifted))
        phase = np.angle(freq_shifted)
        return mag, phase
        
    mag_r, phase_r = get_plot_data(r_glob)
    mag_g, phase_g = get_plot_data(g_glob)
    mag_b, phase_b = get_plot_data(b_glob)
    
    # Re-calculate the hardware weights just for the plot
    N = 64
    p = np.arange(N).reshape(N, 1)
    q = np.arange(N).reshape(1, N)
    W_idft = np.exp(1j * 2 * np.pi * p * q / N)
    W_mapped = np.zeros((N, N))
    W_mapped[:, :N//2] = np.real(W_idft[:, :N//2])
    W_mapped[:, N//2:] = np.imag(W_idft[:, :N//2])
    W_target_plot = (W_mapped + 1.0) / 2.0
    
    # ==========================================
    # 3. THE 14-PANEL DASHBOARD
    # ==========================================
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("MolMem Transient Physics Reconstruction Dashboard", fontsize=22, fontweight='bold')
    
    # --- Row 1: The Pipeline (High-Res -> Low-Res -> Hardware Output) ---
    ax_true = plt.subplot2grid((5, 3), (0, 0))
    ax_true.imshow(true_orig)
    ax_true.set_title("1. True High-Res Original")
    ax_true.axis('off')
    
    ax_reduced = plt.subplot2grid((5, 3), (0, 1))
    ax_reduced.imshow(image)
    ax_reduced.set_title(f"2. Reduced Target Input ({image.shape[0]}x{image.shape[1]})")
    ax_reduced.axis('off')
    
    ax_final = plt.subplot2grid((5, 3), (0, 2))
    ax_final.imshow(final_rgb)
    ax_final.set_title("3. MolMem Reconstructed Image")
    ax_final.axis('off')
    
    # --- Row 2: The Hardware Programming ---
    ax_weights = plt.subplot2grid((5, 3), (1, 1))
    im_w = ax_weights.imshow(W_target_plot, cmap='viridis')
    ax_weights.set_title("4. Programmed Hardware Weights [0.0 to 1.0]")
    ax_weights.axis('off')
    plt.colorbar(im_w, ax=ax_weights, fraction=0.046, pad=0.04, label="Normalized Conductance")
    
    # --- Row 3: Color Panes ---
    ax_r = plt.subplot2grid((5, 3), (2, 0))
    ax_r.imshow(r_glob, cmap='Reds')
    ax_r.set_title("5a. Red Spatial Pane")
    ax_r.axis('off')
    
    ax_g = plt.subplot2grid((5, 3), (2, 1))
    ax_g.imshow(g_glob, cmap='Greens')
    ax_g.set_title("5b. Green Spatial Pane")
    ax_g.axis('off')
    
    ax_b = plt.subplot2grid((5, 3), (2, 2))
    ax_b.imshow(b_glob, cmap='Blues')
    ax_b.set_title("5c. Blue Spatial Pane")
    ax_b.axis('off')
    
    # --- Row 4: Frequency Magnitude ---
    ax_mr = plt.subplot2grid((5, 3), (3, 0))
    ax_mr.imshow(mag_r, cmap='magma')
    ax_mr.set_title("6a. Red 1D Freq Magnitude")
    ax_mr.axis('off')
    
    ax_mg = plt.subplot2grid((5, 3), (3, 1))
    ax_mg.imshow(mag_g, cmap='magma')
    ax_mg.set_title("6b. Green 1D Freq Magnitude")
    ax_mg.axis('off')
    
    ax_mb = plt.subplot2grid((5, 3), (3, 2))
    ax_mb.imshow(mag_b, cmap='magma')
    ax_mb.set_title("6c. Blue 1D Freq Magnitude")
    ax_mb.axis('off')
    
    # --- Row 5: Frequency Phase ---
    ax_pr = plt.subplot2grid((5, 3), (4, 0))
    ax_pr.imshow(phase_r, cmap='hsv')
    ax_pr.set_title("7a. Red 1D Freq Phase")
    ax_pr.axis('off')
    
    ax_pg = plt.subplot2grid((5, 3), (4, 1))
    ax_pg.imshow(phase_g, cmap='hsv')
    ax_pg.set_title("7b. Green 1D Freq Phase")
    ax_pg.axis('off')
    
    ax_pb = plt.subplot2grid((5, 3), (4, 2))
    ax_pb.imshow(phase_b, cmap='hsv')
    ax_pb.set_title("7c. Blue 1D Freq Phase")
    ax_pb.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(fig_dir, "figure_7.png"), bbox_inches="tight"); plt.close()
    print("Saved 14-panel transient physics dashboard to Figures/figure_7.png.")
    
    def calculate_image_metrics(original, reconstructed):
        """Calculates MSE, PSNR, and SNR between two image arrays."""
        # Convert images to float64 to prevent 8-bit integer overflow during squaring
        orig = original.astype(np.float64)
        recon = reconstructed.astype(np.float64)
        
        # 1. Mean Squared Error (MSE)
        mse = np.mean((orig - recon) ** 2)
        
        # 2. Peak Signal-to-Noise Ratio (PSNR)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
        # 3. Signal-to-Noise Ratio (SNR)
        signal_power = np.mean(orig ** 2)
        noise_power = mse
        if noise_power == 0:
            snr = float('inf')
        else:
            snr = 10 * np.log10(signal_power / noise_power)
            
        return mse, psnr, snr
        
    # Use the downscaled target image that was actually processed by the hardware
    target_image = image 
    
    print("\n[Step 9] Computing final image reconstruction quality metrics...")
    if target_image.shape == final_rgb.shape:
        mse_val, psnr_val, snr_val = calculate_image_metrics(target_image, final_rgb)
        
        print("\n=== MolMem Hardware Reconstruction Metrics (Reduced Image) ===")
        print(f"Mean Squared Error (MSE)  : {mse_val:.4f}")
        print(f"Peak SNR (PSNR)           : {psnr_val:.2f} dB")
        print(f"Signal-to-Noise (SNR)     : {snr_val:.2f} dB")
        
    else:
        print(f"[!] Dimension Mismatch! Target is {target_image.shape}, Reconstructed is {final_rgb.shape}.")
    print("\nAll steps completed successfully!")


# In[ ]:





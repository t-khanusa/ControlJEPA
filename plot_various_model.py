import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define Data ---

# Categories on X-axis
categories = ['Llama3', 'Gemma2', 'OpenELM', 'Qwen3', 'R1-Distill', 'OLMo']

# Data for original series (Accuracy % and Error)
# Estimated from the second provided image
data_ntp = [57.5, 33.5, 12.0, 63.0, 52.0, 88.5]
err_ntp = [5.5, 3.5, 2.0, 1.0, 2.0, 0.5]

data_ntp_jepa = [71.5, 43.5, 25.5, 63.5, 54.5, 89.0]
err_ntp_jepa = [1.5, 3.0, 2.5, 0.8, 1.0, 0.5]

data_ntp_stp = [84.5, 57.5, 39.5, 63.2, 65.5, 89.0]
err_ntp_stp = [0.5, 12.0, 12.0, 1.0, 1.0, 0.5]

# Data for the new series (Loss Lyapunov and Error)
# Estimated to be slightly higher than the orange bar, as depicted in the generation
data_lyapunov = [86.5, 61.0, 44.0, 64.0, 69.0, 90.5]
err_lyapunov = [2.5, 11.0, 11.0, 2.5, 3.5, 2.0]

# --- 2. Create the Plot ---

# Positions for the bar groups
x = np.arange(len(categories))

# Width of each bar
width = 0.20

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 7), dpi=100)

# --- 3. Add Bars and Error Bars ---

# Series 1: L_NTP (Blue)
rects1 = ax.bar(x - 1.5*width, data_ntp, width, yerr=err_ntp, capsize=4, color='#4e84b4')

# Series 2: L_NTP + L_JEPA (Green)
rects2 = ax.bar(x - 0.5*width, data_ntp_jepa, width, yerr=err_ntp_jepa, capsize=4, color='#3cb371')

# Series 3: L_NTP + L_STP (Orange)
rects3 = ax.bar(x + 0.5*width, data_ntp_stp, width, yerr=err_ntp_stp, capsize=4, color='#ff8c00')

# Series 4: Loss Lyapunov (Purple)
rects4 = ax.bar(x + 1.5*width, data_lyapunov, width, yerr=err_lyapunov, capsize=4, color='#8e44ad')

# --- 4. Customize Aesthetics and Labels ---

# Increased font sizes
label_fontsize = 24
tick_fontsize = 20
legend_fontsize = 18

# Set Y-axis label and font size
ax.set_ylabel('Accuracy (%)', fontsize=label_fontsize)

# Set X-axis ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=tick_fontsize)

# Add the bottom label matching the original image style
ax.set_xlabel('\n(b) Model families', fontsize=label_fontsize)

# Set Y-axis limits and tick spacing (extended to 100 to fit the new taller bars)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.tick_params(axis='y', labelsize=tick_fontsize)
ax.tick_params(axis='x', labelsize=tick_fontsize)

# Set Legend
ax.legend(loc='upper left', fontsize=legend_fontsize, frameon=True)

# Clean layout
plt.tight_layout()

plt.savefig('plot_various_model2.png', dpi=300)
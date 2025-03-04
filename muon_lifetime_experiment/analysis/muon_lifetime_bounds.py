#!/usr/bin/env python3
"""
Plot Muon Lifetime Data with ROI Fit and Purple Tinted Region

This script reads muon lifetime data (0–10 µs) from a CSV file, fits an exponential decay 
only to the region of interest (ROI: 1.0–4.7 µs), and produces a plot that:
  - Shades the ROI with a purple tint,
  - Colors data in 0–1.0 µs and 4.7–10 µs in gray,
  - Colors data in 1.0–4.7 µs in purple,
  - Overlays the exponential fit (only over the ROI),
  - Displays the fitted parameters in a text box.
  
The seaborn style and Purples palette are used to match your previous work.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Configure logging and seaborn theme
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk", palette="Purples")

def read_csv_file(csv_path: str, separators=None) -> pd.DataFrame:
    """Read and validate a CSV file with two columns: Time and Counts."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    separators = separators or [',', '\t', ';']
    for sep in separators:
        try:
            df = pd.read_csv(csv_path, sep=sep, header=None, names=["Time", "Counts"])
            if df["Counts"].notna().sum() > 0:
                logger.info(f"Successfully read CSV using separator '{sep}'")
                return df.astype({"Time": float, "Counts": float})
        except Exception as e:
            logger.debug(f"Failed with separator '{sep}': {e}")
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python', header=None, names=["Time", "Counts"])
        logger.info("Successfully read CSV by auto-detecting separator")
        return df.astype({"Time": float, "Counts": float})
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

def exponential_decay(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """Exponential decay model: A * exp(-t/tau) + C."""
    return A * np.exp(-t / tau) + C

def main() -> None:
    # Set the data file path (adjust as needed)
    data_path = ("/Users/bekheet/Documents/*Eng. Physics/Winter 2025/453 - Advanced Physics Laboratory/"
                 "453-ADV-PHYS-LAB/muon_lifetime_experiment/analysis/data_files/feb4data_muonlifetime.csv")
    try:
        data = read_csv_file(data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Define the region of interest (ROI) for fitting: 1.0–4.7 µs.
    roi_lower, roi_upper = 1.0, 4.7

    # Filter data for the ROI to perform the fit.
    mask_roi = (data["Time"] >= roi_lower) & (data["Time"] <= roi_upper)
    data_roi = data[mask_roi]
    if data_roi.empty:
        logger.error("No data found in the ROI for fitting.")
        return

    t_fit = data_roi["Time"].values
    counts_fit = data_roi["Counts"].values

    # Initial guess: use max, typical tau (2.2), and min from the ROI.
    p0 = [np.max(counts_fit), 2.2, np.min(counts_fit)]
    try:
        popt, pcov = curve_fit(exponential_decay, t_fit, counts_fit, p0=p0)
    except Exception as e:
        logger.error(f"Exponential fit failed: {e}")
        return
    perr = np.sqrt(np.diag(pcov))
    logger.info(f"Fit results: Amplitude = {popt[0]:.2f} ± {perr[0]:.2f}, "
                f"τ (Time Constant) = {popt[1]:.2f} ± {perr[1]:.2f} µs, "
                f"Background Counts = {popt[2]:.2f} ± {perr[2]:.2f}")

    # Prepare the full dataset (assumed to span 0–10 µs)
    t_all = data["Time"].values
    counts_all = data["Counts"].values

    # Create masks for the three segments:
    mask_before = t_all < roi_lower   # 0 to 1.0 µs → gray
    mask_roi_all = (t_all >= roi_lower) & (t_all <= roi_upper)  # 1.0 to 4.7 µs → purple
    mask_after = t_all > roi_upper  # 4.7 to 10 µs → gray

    # Begin plotting
    plt.figure(figsize=(14, 8))
    
    # Add a purple-tinted rectangular shading over the ROI.
    plt.axvspan(roi_lower, roi_upper, color='#7571B7', alpha=0.1)
    
    # Plot data points:
    plt.scatter(t_all[mask_before], counts_all[mask_before],
                color='gray', s=60, edgecolor='black', label="Outside ROI")
    plt.scatter(t_all[mask_roi_all], counts_all[mask_roi_all],
                color='#7571B7', s=60, edgecolor='black', label="ROI (1.0–4.7 µs)")
    plt.scatter(t_all[mask_after], counts_all[mask_after],
                color='gray', s=60, edgecolor='black', label="Outside ROI")
    
    # Generate and plot the fitted exponential curve over the ROI.
    t_curve = np.linspace(roi_lower, roi_upper, 500)
    counts_curve = exponential_decay(t_curve, *popt)
    plt.plot(t_curve, counts_curve, color='#AAA9D4', linewidth=2, label="Exponential Fit")
    
    # Configure plot appearance
    plt.title(f"Muon Lifetime Data with Exponential Fit in Region of Interest (ROI): {roi_lower}–{roi_upper} µs", fontsize=16)
    plt.xlabel("Time (µs)", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.legend(fontsize=14)
    
    # Add a text box with the fitted parameters (fit computed using the ROI)
    fit_text = (f"Amplitude                 = {popt[0]:.2f} ± {perr[0]:.2f}\n"
                f"τ (Time Constant)     = {popt[1]:.2f} ± {perr[1]:.2f} µs\n"
                f"Background Counts  = {popt[2]:.2f} ± {perr[2]:.2f}")
    plt.text(0.70, 0.85, fit_text, transform=plt.gca().transAxes,
             fontsize=16, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.3))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Semiconductor Bandgap Analysis

This script analyzes temperature-dependent resistivity data to determine 
the bandgap energy of semiconductors (initially Silicon).

It provides various analysis modes:
    - Basic Analysis: Fits resistivity vs temperature data to extract bandgap energy.
    - Intrinsic Region Analysis: Focuses on high-temperature data where R ~ exp(Eg/2kBT)/(T^3/2).
    - Extrinsic Region Analysis: Analyzes low-temperature data where R ~ T^z.
    - Diagnostics: Runs comprehensive diagnostic analyses.
    - All Analyses: Runs multiple analyses sequentially.
    - Compare Two Temperature Ranges: Compares fits from two user-specified ranges.

Usage:
    python bandgap_analysis.py
"""

import argparse
import ast
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import constants

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Configure plotting style
plt.style.use('default')

# Physical constants
kB = constants.Boltzmann  # J/K
eV = constants.electron_volt  # J
kB_eV = kB / eV  # eV/K


def read_csv_file(csv_path: str, separators: Optional[List[str]] = None) -> pd.DataFrame:
    """Read and validate CSV data file."""
    if not os.path.isfile(csv_path):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files")
        csv_path = os.path.join(data_dir, os.path.basename(csv_path))
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    separators = separators or [',', '\t', ';']
    
    for sep in separators:
        try:
            df = pd.read_csv(csv_path, sep=sep, header=None, names=["Temperature", "Resistance"])
            if df["Resistance"].notna().sum() > 0:
                logger.info(f"Successfully read CSV using separator '{sep}'")
                return df.astype({"Temperature": float, "Resistance": float})
        except Exception as e:
            logger.debug(f"Failed with separator '{sep}': {e}")
    
    # Try auto-detection as last resort
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python', header=None, names=["Temperature", "Resistance"])
        logger.info("Successfully read CSV by auto-detecting separator")
        return df.astype({"Temperature": float, "Resistance": float})
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")


def intrinsic_region_model(T: np.ndarray, A: float, Eg: float, C: float) -> np.ndarray:
    """
    Intrinsic region model: A * exp(Eg/2kBT) / (T^3/2) + C
    
    Where:
    - T is temperature in Kelvin
    - Eg is bandgap energy in eV
    - kB is Boltzmann constant in eV/K
    - A is a proportionality constant
    - C is a constant offset
    """
    return A * np.exp(Eg/(2*kB_eV*T)) / (T**(3/2)) + C


def linearized_intrinsic_model(x: np.ndarray, m: float, b: float) -> np.ndarray:
    """
    Linearized intrinsic model for ln(R*T^(3/2)) vs 1/T plot
    y = m*x + b where:
    - x = 1/T
    - y = ln(R*T^(3/2))
    - m = Eg/(2*kB_eV)
    - b = ln(A)
    """
    return m * x + b


def extrinsic_region_model(T: np.ndarray, B: float, z: float) -> np.ndarray:
    """
    Extrinsic region model: B * T^z
    
    Where:
    - T is temperature in Kelvin
    - B is related to majority carrier concentration
    - z reflects mobility temperature dependence
    """
    return B * (T**z)


def filter_temperature_range(data: pd.DataFrame, T_min: float, T_max: float) -> pd.DataFrame:
    """Filter data to specified temperature range."""
    filtered = data[(data["Temperature"] >= T_min) & (data["Temperature"] <= T_max)]
    logger.info(f"Filtered to {T_min}–{T_max} K. {len(filtered)} points remain.")
    return filtered


def fit_intrinsic_region(
    T_data: np.ndarray,
    R_data: np.ndarray,
    p0: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit intrinsic region model to data."""
    if p0 is None:
        # Initial guess: A=1, Eg=1.1 eV, C=0
        p0 = [1.0, 1.1, 0.0]
    
    popt, pcov = curve_fit(intrinsic_region_model, T_data, R_data, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def fit_linearized_intrinsic(
    T_data: np.ndarray,
    R_data: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Fit linearized intrinsic model to extract bandgap.
    
    Returns:
    - Eg: Bandgap energy in eV
    - Eg_err: Error in bandgap energy in eV
    - A: Proportionality constant
    - A_err: Error in proportionality constant
    """
    x = 1/T_data
    y = np.log(R_data * (T_data**(3/2)))
    
    # Linear fit: y = mx + b
    popt, pcov = np.polyfit(x, y, 1, cov=True)
    m, b = popt
    perr = np.sqrt(np.diag(pcov))
    m_err, b_err = perr
    
    # Extract physical parameters
    Eg = 2 * kB_eV * m
    Eg_err = 2 * kB_eV * m_err
    A = np.exp(b)
    A_err = A * b_err
    
    return Eg, Eg_err, A, A_err


def fit_extrinsic_region(
    T_data: np.ndarray,
    R_data: np.ndarray,
    p0: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit extrinsic region model to data."""
    if p0 is None:
        # Initial guess: B=R_data[0]/(T_data[0]**1.5), z=1.5
        p0 = [R_data[0]/(T_data[0]**1.5), 1.5]
    
    popt, pcov = curve_fit(extrinsic_region_model, T_data, R_data, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def plot_intrinsic_results(
    T_data: np.ndarray,
    R_data: np.ndarray,
    popt: np.ndarray,
    perr: np.ndarray,
    bounds: Tuple[float, float],
    save_fig: bool = False,
    prefix: str = "bandgap"
) -> None:
    """Plot intrinsic region fit results."""
    # Generate fit curve
    T_fit = np.linspace(float(np.min(T_data)), float(np.max(T_data)), 500)
    R_fit = intrinsic_region_model(T_fit, *popt)
    residuals = R_data - intrinsic_region_model(T_data, *popt)

    # Main fit plot
    plt.figure(figsize=(14, 8))
    plt.scatter(T_data, R_data, label='Data', alpha=0.6)
    plt.plot(T_fit, R_fit, 'r-', label='Fit')
    
    plt.title(f"Semiconductor Resistance vs Temperature ({bounds[0]}–{bounds[1]} K)", fontsize=16)
    plt.xlabel("Temperature (K)", fontsize=14)
    plt.ylabel("Resistance (Ohms)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Add fit parameters text box
    Eg = popt[1]
    Eg_err = perr[1]
    fit_text = (
        f"A = {popt[0]:.2e} ± {perr[0]:.2e}\n"
        f"Eg = {Eg:.4f} ± {Eg_err:.4f} eV\n"
        f"C = {popt[2]:.2e} ± {perr[2]:.2e}"
    )
    plt.text(
        0.80, 0.85, fit_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    if save_fig:
        fit_filename = f"{prefix}_intrinsic_{bounds[0]}_{bounds[1]}_fit.png"
        plt.savefig(fit_filename, bbox_inches="tight")
        logger.info(f"Saved fit plot: {fit_filename}")

    plt.tight_layout()
    plt.show()

    # Residuals plot
    plt.figure(figsize=(14, 4))
    plt.scatter(T_data, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Residuals", fontsize=16)
    plt.xlabel("Temperature (K)", fontsize=14)
    plt.ylabel("Residuals (Ohms)", fontsize=14)
    plt.grid(True)

    if save_fig:
        res_filename = f"{prefix}_intrinsic_{bounds[0]}_{bounds[1]}_residuals.png"
        plt.savefig(res_filename, bbox_inches="tight")
        logger.info(f"Saved residuals plot: {res_filename}")

    plt.tight_layout()
    plt.show()
    
    # Linearized plot for bandgap extraction
    plt.figure(figsize=(14, 8))
    x = 1/T_data
    y = np.log(R_data * (T_data**(3/2)))
    
    plt.scatter(x, y, label='Linearized Data', alpha=0.6)
    
    # Fit line
    Eg, Eg_err, A, A_err = fit_linearized_intrinsic(T_data, R_data)
    x_fit = np.linspace(min(x), max(x), 100)
    m = Eg / (2 * kB_eV)
    b = np.log(A)
    y_fit = linearized_intrinsic_model(x_fit, m, b)
    
    plt.plot(x_fit, y_fit, 'r-', label='Linear Fit')
    
    plt.title(f"Linearized Plot for Bandgap Extraction ({bounds[0]}–{bounds[1]} K)", fontsize=16)
    plt.xlabel("1/T (K$^{-1}$)", fontsize=14)
    plt.ylabel(r"ln(R·T$^{3/2}$)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Add fit parameters text box
    linear_fit_text = (
        f"Slope = {m:.2f}\n"
        f"Eg = {Eg:.4f} ± {Eg_err:.4f} eV\n"
        f"A = {A:.2e} ± {A_err:.2e}"
    )
    plt.text(
        0.80, 0.85, linear_fit_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    if save_fig:
        lin_filename = f"{prefix}_linearized_{bounds[0]}_{bounds[1]}.png"
        plt.savefig(lin_filename, bbox_inches="tight")
        logger.info(f"Saved linearized plot: {lin_filename}")

    plt.tight_layout()
    plt.show()


def plot_extrinsic_results(
    T_data: np.ndarray,
    R_data: np.ndarray,
    popt: np.ndarray,
    perr: np.ndarray,
    bounds: Tuple[float, float],
    save_fig: bool = False,
    prefix: str = "bandgap"
) -> None:
    """Plot extrinsic region fit results."""
    # Generate fit curve
    T_fit = np.linspace(float(np.min(T_data)), float(np.max(T_data)), 500)
    R_fit = extrinsic_region_model(T_fit, *popt)
    residuals = R_data - extrinsic_region_model(T_data, *popt)

    # Main fit plot
    plt.figure(figsize=(14, 8))
    plt.scatter(T_data, R_data, label='Data', alpha=0.6)
    plt.plot(T_fit, R_fit, 'g-', label='Fit')
    
    plt.title(f"Extrinsic Region Analysis ({bounds[0]}–{bounds[1]} K)", fontsize=16)
    plt.xlabel("Temperature (K)", fontsize=14)
    plt.ylabel("Resistance (Ohms)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Add fit parameters text box
    B, z = popt
    B_err, z_err = perr
    fit_text = (
        f"B = {B:.2e} ± {B_err:.2e}\n"
        f"z = {z:.4f} ± {z_err:.4f}"
    )
    plt.text(
        0.80, 0.85, fit_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    if save_fig:
        fit_filename = f"{prefix}_extrinsic_{bounds[0]}_{bounds[1]}_fit.png"
        plt.savefig(fit_filename, bbox_inches="tight")
        logger.info(f"Saved fit plot: {fit_filename}")

    plt.tight_layout()
    plt.show()

    # Residuals plot
    plt.figure(figsize=(14, 4))
    plt.scatter(T_data, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Residuals", fontsize=16)
    plt.xlabel("Temperature (K)", fontsize=14)
    plt.ylabel("Residuals (Ohms)", fontsize=14)
    plt.grid(True)

    if save_fig:
        res_filename = f"{prefix}_extrinsic_{bounds[0]}_{bounds[1]}_residuals.png"
        plt.savefig(res_filename, bbox_inches="tight")
        logger.info(f"Saved residuals plot: {res_filename}")

    plt.tight_layout()
    plt.show()
    
    # Log-log plot for extrinsic region
    plt.figure(figsize=(14, 8))
    plt.loglog(T_data, R_data, 'o', label='Data')
    plt.loglog(T_fit, R_fit, 'g-', label='Fit')
    
    plt.title(f"Log-Log Plot for Extrinsic Region ({bounds[0]}–{bounds[1]} K)", fontsize=16)
    plt.xlabel("Temperature (K)", fontsize=14)
    plt.ylabel("Resistance (Ohms)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both")

    if save_fig:
        loglog_filename = f"{prefix}_extrinsic_loglog_{bounds[0]}_{bounds[1]}.png"
        plt.savefig(loglog_filename, bbox_inches="tight")
        logger.info(f"Saved log-log plot: {loglog_filename}")

    plt.tight_layout()
    plt.show()


def process_intrinsic_region(
    data: pd.DataFrame,
    bounds_list: List[Tuple[float, float]],
    save_plots: bool = False
) -> List[Dict[str, Any]]:
    """Process data for intrinsic region analysis with multiple temperature bounds."""
    results = []

    for lb, ub in bounds_list:
        logger.info(f"\nAnalyzing intrinsic region in temperature range: {lb}–{ub} K")
        
        # Filter data
        filtered = filter_temperature_range(data, lb, ub)
        if filtered.empty:
            logger.warning("No data in range")
            continue

        # Fit and plot
        try:
            T_data = filtered["Temperature"].values
            R_data = filtered["Resistance"].values
            popt, perr = fit_intrinsic_region(T_data, R_data)
            
            logger.info(
                f"Intrinsic fit results: A={popt[0]:.2e}±{perr[0]:.2e}, "
                f"Eg={popt[1]:.4f}±{perr[1]:.4f} eV, "
                f"C={popt[2]:.2e}±{perr[2]:.2e}"
            )
            
            # Also get linearized fit results
            Eg, Eg_err, A, A_err = fit_linearized_intrinsic(T_data, R_data)
            logger.info(
                f"Linearized fit results: Eg={Eg:.4f}±{Eg_err:.4f} eV, "
                f"A={A:.2e}±{A_err:.2e}"
            )

            results.append({
                "bounds": (lb, ub),
                "popt": popt,
                "perr": perr,
                "T_data": T_data,
                "R_data": R_data,
                "linearized_results": (Eg, Eg_err, A, A_err)
            })

            plot_intrinsic_results(T_data, R_data, popt, perr, (lb, ub), save_plots)
            
        except RuntimeError as e:
            logger.error(f"Fit failed: {e}")
            continue

    return results


def process_extrinsic_region(
    data: pd.DataFrame,
    bounds_list: List[Tuple[float, float]],
    save_plots: bool = False
) -> List[Dict[str, Any]]:
    """Process data for extrinsic region analysis with multiple temperature bounds."""
    results = []

    for lb, ub in bounds_list:
        logger.info(f"\nAnalyzing extrinsic region in temperature range: {lb}–{ub} K")
        
        # Filter data
        filtered = filter_temperature_range(data, lb, ub)
        if filtered.empty:
            logger.warning("No data in range")
            continue

        # Fit and plot
        try:
            T_data = filtered["Temperature"].values
            R_data = filtered["Resistance"].values
            popt, perr = fit_extrinsic_region(T_data, R_data)
            
            logger.info(
                f"Extrinsic fit results: B={popt[0]:.2e}±{perr[0]:.2e}, "
                f"z={popt[1]:.4f}±{perr[1]:.4f}"
            )

            results.append({
                "bounds": (lb, ub),
                "popt": popt,
                "perr": perr,
                "T_data": T_data,
                "R_data": R_data
            })

            plot_extrinsic_results(T_data, R_data, popt, perr, (lb, ub), save_plots)
            
        except RuntimeError as e:
            logger.error(f"Fit failed: {e}")
            continue

    return results


def search_transition_temperature(
    data: pd.DataFrame,
    n_steps: int = 20,
    verbose: bool = True,
    plot_result: bool = True
) -> Optional[float]:
    """
    Search for the transition temperature between extrinsic and intrinsic regions.
    
    Returns:
    - Transition temperature in K
    """
    T_min, T_max = data["Temperature"].min(), data["Temperature"].max()
    T_range = np.linspace(T_min + 0.1*(T_max-T_min), T_max - 0.1*(T_max-T_min), n_steps)
    
    best_transition = None
    min_total_error = float('inf')
    
    for T_trans in T_range:
        # Extrinsic region (lower temperatures)
        extrinsic_data = data[data["Temperature"] <= T_trans]
        # Intrinsic region (higher temperatures)
        intrinsic_data = data[data["Temperature"] >= T_trans]
        
        if len(extrinsic_data) < 5 or len(intrinsic_data) < 5:
            continue
        
        # Fit extrinsic region
        try:
            T_extrinsic = extrinsic_data["Temperature"].values
            R_extrinsic = extrinsic_data["Resistance"].values
            popt_extrinsic, _ = fit_extrinsic_region(T_extrinsic, R_extrinsic)
            extrinsic_error = np.mean((R_extrinsic - extrinsic_region_model(T_extrinsic, *popt_extrinsic))**2)
        except RuntimeError:
            continue
            
        # Fit intrinsic region
        try:
            T_intrinsic = intrinsic_data["Temperature"].values
            R_intrinsic = intrinsic_data["Resistance"].values
            popt_intrinsic, _ = fit_intrinsic_region(T_intrinsic, R_intrinsic)
            intrinsic_error = np.mean((R_intrinsic - intrinsic_region_model(T_intrinsic, *popt_intrinsic))**2)
        except RuntimeError:
            continue
        
        # Total error (normalized)
        total_error = extrinsic_error/np.mean(R_extrinsic)**2 + intrinsic_error/np.mean(R_intrinsic)**2
        
        if verbose:
            logger.info(
                f"Transition T={T_trans:.1f} K: "
                f"Extrinsic error={extrinsic_error:.2e}, "
                f"Intrinsic error={intrinsic_error:.2e}, "
                f"Total={total_error:.2e}"
            )
        
        if total_error < min_total_error:
            min_total_error = total_error
            best_transition = T_trans
    
    if best_transition is None:
        logger.error("Could not find a valid transition temperature")
        return None
    
    logger.info(f"Found transition temperature: {best_transition:.1f} K")
    
    if plot_result and best_transition is not None:
        # Plot the data with the transition temperature
        plt.figure(figsize=(14, 8))
        plt.scatter(data["Temperature"], data["Resistance"], label='Data', alpha=0.6)
        plt.axvline(best_transition, color='r', linestyle='--', label=f'Transition: {best_transition:.1f} K')
        
        # Fit and plot extrinsic region
        extrinsic_data = data[data["Temperature"] <= best_transition]
        T_extrinsic = extrinsic_data["Temperature"].values
        R_extrinsic = extrinsic_data["Resistance"].values
        popt_extrinsic, _ = fit_extrinsic_region(T_extrinsic, R_extrinsic)
        
        T_fit_extrinsic = np.linspace(T_min, best_transition, 100)
        R_fit_extrinsic = extrinsic_region_model(T_fit_extrinsic, *popt_extrinsic)
        plt.plot(T_fit_extrinsic, R_fit_extrinsic, 'g-', label=f'Extrinsic Fit (z={popt_extrinsic[1]:.3f})')
        
        # Fit and plot intrinsic region
        intrinsic_data = data[data["Temperature"] >= best_transition]
        T_intrinsic = intrinsic_data["Temperature"].values
        R_intrinsic = intrinsic_data["Resistance"].values
        popt_intrinsic, _ = fit_intrinsic_region(T_intrinsic, R_intrinsic)
        
        T_fit_intrinsic = np.linspace(best_transition, T_max, 100)
        R_fit_intrinsic = intrinsic_region_model(T_fit_intrinsic, *popt_intrinsic)
        plt.plot(T_fit_intrinsic, R_fit_intrinsic, 'r-', label=f'Intrinsic Fit (Eg={popt_intrinsic[1]:.3f} eV)')
        
        plt.title("Transition Between Extrinsic and Intrinsic Regions", fontsize=16)
        plt.xlabel("Temperature (K)", fontsize=14)
        plt.ylabel("Resistance (Ohms)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return best_transition


def run_diagnostics(
    data: pd.DataFrame
) -> Dict[str, Any]:
    """Run comprehensive diagnostic analysis for semiconductor data."""
    logger.info("\nRunning diagnostic analysis...")
    T_data = data["Temperature"].values
    R_data = data["Resistance"].values
    diagnostics: Dict[str, Any] = {}

    # Check for temperature range
    T_min, T_max = np.min(T_data), np.max(T_data)
    T_range = T_max - T_min
    logger.info(f"Temperature range: {T_min:.1f} - {T_max:.1f} K (span: {T_range:.1f} K)")
    diagnostics["temperature_range"] = (T_min, T_max)
    
    if T_max < 500:
        logger.warning("Maximum temperature may be too low to observe intrinsic behavior")
    
    # Check data density
    point_density = len(T_data) / T_range
    logger.info(f"Data point density: {point_density:.2f} points/K")
    diagnostics["point_density"] = point_density
    
    if point_density < 1:
        logger.warning("Low data density may affect fit quality")
    
    # Estimate transition temperature
    try:
        transition_T = search_transition_temperature(data, plot_result=True)
        diagnostics["transition_temperature"] = transition_T
    except Exception as e:
        logger.error(f"Failed to estimate transition temperature: {e}")
    
    # Estimate bandgap from highest temperature data
    try:
        high_temp_data = data[data["Temperature"] > np.percentile(T_data, 70)]
        if len(high_temp_data) >= 5:
            T_high = high_temp_data["Temperature"].values
            R_high = high_temp_data["Resistance"].values
            Eg, Eg_err, _, _ = fit_linearized_intrinsic(T_high, R_high)
            logger.info(f"Estimated bandgap from high-T data: {Eg:.4f} ± {Eg_err:.4f} eV")
            diagnostics["high_temp_bandgap"] = (Eg, Eg_err)
        else:
            logger.warning("Not enough high-temperature data points for bandgap estimation")
    except Exception as e:
        logger.error(f"Failed to estimate bandgap from high-temp data: {e}")
    
    # Estimate extrinsic parameter from low temperature data
    try:
        low_temp_data = data[data["Temperature"] < np.percentile(T_data, 30)]
        if len(low_temp_data) >= 5:
            T_low = low_temp_data["Temperature"].values
            R_low = low_temp_data["Resistance"].values
            popt, perr = fit_extrinsic_region(T_low, R_low)
            logger.info(f"Estimated extrinsic parameters: B={popt[0]:.2e}, z={popt[1]:.4f} ± {perr[1]:.4f}")
            diagnostics["extrinsic_params"] = (popt, perr)
        else:
            logger.warning("Not enough low-temperature data points for extrinsic parameter estimation")
    except Exception as e:
        logger.error(f"Failed to estimate extrinsic parameters: {e}")
    
    # Check for data anomalies
    R_sorted = np.sort(R_data)
    R_diff = np.diff(R_sorted)
    anomalies = np.where(R_diff > 5 * np.median(R_diff))[0]
    if len(anomalies) > 0:
        logger.warning(f"Possible anomalies detected at indices: {anomalies}")
        diagnostics["anomalies"] = anomalies
    
    # Plot diagnostics
    plt.figure(figsize=(14, 8))
    plt.semilogy(1000/T_data, R_data, 'o-')
    plt.title("Arrhenius Plot", fontsize=16)
    plt.xlabel("1000/T (1000/K)", fontsize=14)
    plt.ylabel("Resistance (Ohms)", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return diagnostics


def get_data_files() -> List[str]:
    """Get list of CSV files in the data_files directory."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created data_files directory at {data_dir}")
        return []
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    return csv_files


def select_file() -> Optional[str]:
    """Interactive file selection."""
    csv_files = get_data_files()
    
    if not csv_files:
        logger.error("No CSV files found in data_files directory!")
        logger.info("Please place your data files in the 'data_files' directory.")
        return None
        
    print("\nAvailable data files:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    while True:
        try:
            choice = int(input("\nSelect file number (or 0 to exit): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(csv_files):
                return os.path.join("data_files", csv_files[choice - 1])
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def compare_temperature_ranges(
    data: pd.DataFrame,
    save_plots: bool = False
) -> None:
    """Run comparative analysis for two different temperature ranges."""
    logger.info("\nComparative Temperature Range Analysis")
    
    logger.info("\nEnter first temperature range for analysis:")
    bounds1 = get_bounds_input()
    if not bounds1:
        return

    logger.info("\nEnter second temperature range for analysis:")
    bounds2 = get_bounds_input()
    if not bounds2:
        return
    
    # Ask for analysis type
    analysis_type = input("\nAnalysis type (1-Intrinsic, 2-Extrinsic): ").strip()
    
    if analysis_type == "1":
        # Intrinsic analysis
        results = process_intrinsic_region(
            data,
            [bounds1, bounds2],
            save_plots=False  # We'll create a comparative plot instead
        )
        
        if len(results) == 2:
            # Create comparative plot
            plt.figure(figsize=(14, 10))
            
            # Plot first range
            plt.subplot(2, 1, 1)
            T_data1 = results[0]["T_data"]
            R_data1 = results[0]["R_data"]
            popt1 = results[0]["popt"]
            bounds1 = results[0]["bounds"]
            
            plt.scatter(T_data1, R_data1, label='Data', alpha=0.6)
            T_fit1 = np.linspace(min(T_data1), max(T_data1), 100)
            R_fit1 = intrinsic_region_model(T_fit1, *popt1)
            plt.plot(T_fit1, R_fit1, 'r-', label='Fit')
            
            plt.title(f"Range: {bounds1[0]}–{bounds1[1]} K, Bandgap: {popt1[1]:.4f} eV", fontsize=14)
            plt.xlabel("Temperature (K)", fontsize=12)
            plt.ylabel("Resistance (Ohms)", fontsize=12)
            plt.legend()
            plt.grid(True)
            
            # Plot second range
            plt.subplot(2, 1, 2)
            T_data2 = results[1]["T_data"]
            R_data2 = results[1]["R_data"]
            popt2 = results[1]["popt"]
            bounds2 = results[1]["bounds"]
            
            plt.scatter(T_data2, R_data2, label='Data', alpha=0.6)
            T_fit2 = np.linspace(min(T_data2), max(T_data2), 100)
            R_fit2 = intrinsic_region_model(T_fit2, *popt2)
            plt.plot(T_fit2, R_fit2, 'r-', label='Fit')
            
            plt.title(f"Range: {bounds2[0]}–{bounds2[1]} K, Bandgap: {popt2[1]:.4f} eV", fontsize=14)
            plt.xlabel("Temperature (K)", fontsize=12)
            plt.ylabel("Resistance (Ohms)", fontsize=12)
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f"bandgap_comparative_intrinsic.png", dpi=300, bbox_inches='tight')
                logger.info("Saved comparative plot")
            
            plt.show()
            
            # Linearized comparison
            plt.figure(figsize=(14, 10))
            
            # Plot first range linearized
            plt.subplot(2, 1, 1)
            x1 = 1/T_data1
            y1 = np.log(R_data1 * (T_data1**(3/2)))
            
            plt.scatter(x1, y1, label='Linearized Data', alpha=0.6)
            Eg1, Eg_err1, _, _ = results[0]["linearized_results"]
            
            # Fit line
            x_fit1 = np.linspace(min(x1), max(x1), 100)
            m1 = Eg1 / (2 * kB_eV)
            b1 = np.log(results[0]["popt"][0])
            y_fit1 = linearized_intrinsic_model(x_fit1, m1, b1)
            
            plt.plot(x_fit1, y_fit1, 'r-', label=f'Eg = {Eg1:.4f} ± {Eg_err1:.4f} eV')
            
            plt.title(f"Linearized Plot: {bounds1[0]}–{bounds1[1]} K", fontsize=14)
            plt.xlabel("1/T (K$^{-1}$)", fontsize=12)
            plt.ylabel(r"ln(R·T$^{3/2}$)", fontsize=12)
            plt.legend()
            plt.grid(True)
            
            # Plot second range linearized
            plt.subplot(2, 1, 2)
            x2 = 1/T_data2
            y2 = np.log(R_data2 * (T_data2**(3/2)))
            
            plt.scatter(x2, y2, label='Linearized Data', alpha=0.6)
            Eg2, Eg_err2, _, _ = results[1]["linearized_results"]
            
            # Fit line
            x_fit2 = np.linspace(min(x2), max(x2), 100)
            m2 = Eg2 / (2 * kB_eV)
            b2 = np.log(results[1]["popt"][0])
            y_fit2 = linearized_intrinsic_model(x_fit2, m2, b2)
            
            plt.plot(x_fit2, y_fit2, 'r-', label=f'Eg = {Eg2:.4f} ± {Eg_err2:.4f} eV')
            
            plt.title(f"Linearized Plot: {bounds2[0]}–{bounds2[1]} K", fontsize=14)
            plt.xlabel("1/T (K$^{-1}$)", fontsize=12)
            plt.ylabel(r"ln(R·T$^{3/2}$)", fontsize=12)
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f"bandgap_comparative_linearized.png", dpi=300, bbox_inches='tight')
                logger.info("Saved comparative linearized plot")
            
            plt.show()
        else:
            logger.error("Failed to generate both fits for comparison")
            
    elif analysis_type == "2":
        # Extrinsic analysis
        results = process_extrinsic_region(
            data,
            [bounds1, bounds2],
            save_plots=False
        )
        
        if len(results) == 2:
            # Create comparative plot
            plt.figure(figsize=(14, 10))
            
            # Plot first range
            plt.subplot(2, 1, 1)
            T_data1 = results[0]["T_data"]
            R_data1 = results[0]["R_data"]
            popt1 = results[0]["popt"]
            bounds1 = results[0]["bounds"]
            
            plt.scatter(T_data1, R_data1, label='Data', alpha=0.6)
            T_fit1 = np.linspace(min(T_data1), max(T_data1), 100)
            R_fit1 = extrinsic_region_model(T_fit1, *popt1)
            plt.plot(T_fit1, R_fit1, 'g-', label='Fit')
            
            plt.title(f"Range: {bounds1[0]}–{bounds1[1]} K, z-parameter: {popt1[1]:.4f}", fontsize=14)
            plt.xlabel("Temperature (K)", fontsize=12)
            plt.ylabel("Resistance (Ohms)", fontsize=12)
            plt.legend()
            plt.grid(True)
            
            # Plot second range
            plt.subplot(2, 1, 2)
            T_data2 = results[1]["T_data"]
            R_data2 = results[1]["R_data"]
            popt2 = results[1]["popt"]
            bounds2 = results[1]["bounds"]
            
            plt.scatter(T_data2, R_data2, label='Data', alpha=0.6)
            T_fit2 = np.linspace(min(T_data2), max(T_data2), 100)
            R_fit2 = extrinsic_region_model(T_fit2, *popt2)
            plt.plot(T_fit2, R_fit2, 'g-', label='Fit')
            
            plt.title(f"Range: {bounds2[0]}–{bounds2[1]} K, z-parameter: {popt2[1]:.4f}", fontsize=14)
            plt.xlabel("Temperature (K)", fontsize=12)
            plt.ylabel("Resistance (Ohms)", fontsize=12)
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f"bandgap_comparative_extrinsic.png", dpi=300, bbox_inches='tight')
                logger.info("Saved comparative plot")
            
            plt.show()
            
            # Log-log comparison
            plt.figure(figsize=(14, 10))
            
            # Plot first range log-log
            plt.subplot(2, 1, 1)
            plt.loglog(T_data1, R_data1, 'o', label='Data')
            plt.loglog(T_fit1, R_fit1, 'g-', label=f'z = {popt1[1]:.4f}')
            
            plt.title(f"Log-Log Plot: {bounds1[0]}–{bounds1[1]} K", fontsize=14)
            plt.xlabel("Temperature (K)", fontsize=12)
            plt.ylabel("Resistance (Ohms)", fontsize=12)
            plt.legend()
            plt.grid(True, which="both")
            
            # Plot second range log-log
            plt.subplot(2, 1, 2)
            plt.loglog(T_data2, R_data2, 'o', label='Data')
            plt.loglog(T_fit2, R_fit2, 'g-', label=f'z = {popt2[1]:.4f}')
            
            plt.title(f"Log-Log Plot: {bounds2[0]}–{bounds2[1]} K", fontsize=14)
            plt.xlabel("Temperature (K)", fontsize=12)
            plt.ylabel("Resistance (Ohms)", fontsize=12)
            plt.legend()
            plt.grid(True, which="both")
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f"bandgap_comparative_extrinsic_loglog.png", dpi=300, bbox_inches='tight')
                logger.info("Saved comparative log-log plot")
            
            plt.show()
        else:
            logger.error("Failed to generate both fits for comparison")
    else:
        logger.error("Invalid analysis type")


def get_bounds_input() -> Optional[Tuple[float, float]]:
    """Get temperature bounds input from user."""
    try:
        bounds_str = input("Enter temperature bounds as 'lower,upper' (e.g., '300,600'): ").strip()
        if not bounds_str:
            return None
        lb, ub = map(float, bounds_str.split(','))
        return (lb, ub)
    except ValueError:
        logger.error("Invalid bounds format")
        return None


def print_menu() -> None:
    """Display the main menu."""
    print("\nSemiconductor Bandgap Analysis Menu")
    print("=" * 40)
    print("1. Intrinsic Region Analysis")
    print("2. Extrinsic Region Analysis")
    print("3. Find Transition Temperature")
    print("4. Run Diagnostics")
    print("5. Run All Analyses")
    print("6. Compare Two Temperature Ranges")
    print("7. Exit")
    print("=" * 40)


def main() -> None:
    """Interactive main entry point for semiconductor bandgap analysis."""
    print("\nSemiconductor Bandgap Analysis Tool")
    print("=" * 40)
    print("This tool analyzes temperature-dependent resistivity data")
    print("to determine the bandgap energy of semiconductors.")
    print("=" * 40)
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-7): ").strip()
        if choice == "7":
            print("Exiting program...")
            break
        if choice not in ["1", "2", "3", "4", "5", "6"]:
            print("Invalid choice. Please try again.")
            continue

        # Ask which data file to use
        file_path = select_file()
        if not file_path:
            continue

        # Ask for saving plots
        save_plots_input = input("Save plots to files? (y/n, default: n): ").lower().strip()
        save_plots = save_plots_input.startswith('y')

        # Read data
        try:
            data = read_csv_file(file_path)
            logger.info("\nData Preview:")
            logger.info(data.head().to_string())
        except Exception as e:
            logger.error(f"Failed to read data: {e}")
            input("\nPress Enter to return to main menu...")
            continue

        if choice == "1":
            # Intrinsic Region Analysis
            logger.info("\nRunning intrinsic region analysis...")
            print("\nEnter custom temperature ranges as (lb, ub); (lb2, ub2); etc.")
            print("Example: (600,900);(650,850)")
            custom_bounds_str = input("Press ENTER for default bounds: ").strip()
            
            default_bounds = [
                (600, 900), (650, 850), (700, 900),
                (600, 850), (650, 900), (700, 850)
            ]
            
            if not custom_bounds_str:
                used_bounds = default_bounds
            else:
                used_bounds = []
                for chunk in custom_bounds_str.split(';'):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    try:
                        parsed = ast.literal_eval(chunk)
                        lb_val = float(parsed[0])
                        ub_val = float(parsed[1])
                        used_bounds.append((lb_val, ub_val))
                    except Exception as pe:
                        logger.warning(f"Failed to parse '{chunk}': {pe}")
                if not used_bounds:
                    logger.warning("No valid custom bounds parsed, using defaults.")
                    used_bounds = default_bounds

            results = process_intrinsic_region(
                data,
                used_bounds,
                save_plots=save_plots
            )
            
            if results:
                logger.info("\nIntrinsic Region Analysis Results Summary:")
                for res in results:
                    lb, ub = res["bounds"]
                    popt, perr = res["popt"], res["perr"]
                    Eg, Eg_err, _, _ = res["linearized_results"]
                    logger.info(
                        f"Bounds {lb:.2f}–{ub:.2f} K: "
                        f"Eg(direct)={popt[1]:.4f}±{perr[1]:.4f} eV, "
                        f"Eg(linearized)={Eg:.4f}±{Eg_err:.4f} eV"
                    )
            
        elif choice == "2":
            # Extrinsic Region Analysis
            logger.info("\nRunning extrinsic region analysis...")
            print("\nEnter custom temperature ranges as (lb, ub); (lb2, ub2); etc.")
            print("Example: (200,400);(250,350)")
            custom_bounds_str = input("Press ENTER for default bounds: ").strip()
            
            default_bounds = [
                (200, 400), (250, 350), (300, 400),
                (200, 350), (250, 400), (300, 350)
            ]
            
            if not custom_bounds_str:
                used_bounds = default_bounds
            else:
                used_bounds = []
                for chunk in custom_bounds_str.split(';'):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    try:
                        parsed = ast.literal_eval(chunk)
                        lb_val = float(parsed[0])
                        ub_val = float(parsed[1])
                        used_bounds.append((lb_val, ub_val))
                    except Exception as pe:
                        logger.warning(f"Failed to parse '{chunk}': {pe}")
                if not used_bounds:
                    logger.warning("No valid custom bounds parsed, using defaults.")
                    used_bounds = default_bounds

            results = process_extrinsic_region(
                data,
                used_bounds,
                save_plots=save_plots
            )
            
            if results:
                logger.info("\nExtrinsic Region Analysis Results Summary:")
                for res in results:
                    lb, ub = res["bounds"]
                    popt, perr = res["popt"], res["perr"]
                    logger.info(
                        f"Bounds {lb:.2f}–{ub:.2f} K: "
                        f"B={popt[0]:.2e}±{perr[0]:.2e}, "
                        f"z={popt[1]:.4f}±{perr[1]:.4f}"
                    )
            
        elif choice == "3":
            # Find Transition Temperature
            logger.info("\nSearching for transition temperature...")
            transition_T = search_transition_temperature(data, verbose=True, plot_result=True)
            
            if transition_T is not None:
                logger.info(f"\nTransition temperature: {transition_T:.1f} K")
                
                # Analyze regions separately
                logger.info("\nAnalyzing extrinsic region (below transition)...")
                extrinsic_data = data[data["Temperature"] <= transition_T]
                T_extrinsic = extrinsic_data["Temperature"].values
                R_extrinsic = extrinsic_data["Resistance"].values
                
                try:
                    popt_extrinsic, perr_extrinsic = fit_extrinsic_region(T_extrinsic, R_extrinsic)
                    logger.info(
                        f"Extrinsic parameters: B={popt_extrinsic[0]:.2e}±{perr_extrinsic[0]:.2e}, "
                        f"z={popt_extrinsic[1]:.4f}±{perr_extrinsic[1]:.4f}"
                    )
                except RuntimeError as e:
                    logger.error(f"Extrinsic fit failed: {e}")
                
                logger.info("\nAnalyzing intrinsic region (above transition)...")
                intrinsic_data = data[data["Temperature"] >= transition_T]
                T_intrinsic = intrinsic_data["Temperature"].values
                R_intrinsic = intrinsic_data["Resistance"].values
                
                try:
                    popt_intrinsic, perr_intrinsic = fit_intrinsic_region(T_intrinsic, R_intrinsic)
                    Eg, Eg_err, A, A_err = fit_linearized_intrinsic(T_intrinsic, R_intrinsic)
                    
                    logger.info(
                        f"Intrinsic parameters (direct): A={popt_intrinsic[0]:.2e}±{perr_intrinsic[0]:.2e}, "
                        f"Eg={popt_intrinsic[1]:.4f}±{perr_intrinsic[1]:.4f} eV"
                    )
                    logger.info(
                        f"Intrinsic parameters (linearized): Eg={Eg:.4f}±{Eg_err:.4f} eV, "
                        f"A={A:.2e}±{A_err:.2e}"
                    )
                except RuntimeError as e:
                    logger.error(f"Intrinsic fit failed: {e}")
            
        elif choice == "4":
            # Diagnostics
            logger.info("\nRunning diagnostic analysis...")
            diagnostics = run_diagnostics(data)
            
            logger.info("\nDiagnostic Results Summary:")
            print("\nTemperature Range:", diagnostics.get("temperature_range", "N/A"))
            print("Data Point Density:", diagnostics.get("point_density", "N/A"), "points/K")
            
            if "transition_temperature" in diagnostics:
                print(f"Estimated Transition Temperature: {diagnostics['transition_temperature']:.1f} K")
            
            if "high_temp_bandgap" in diagnostics:
                Eg, Eg_err = diagnostics["high_temp_bandgap"]
                print(f"Estimated Bandgap: {Eg:.4f} ± {Eg_err:.4f} eV")
            
            if "extrinsic_params" in diagnostics:
                popt, perr = diagnostics["extrinsic_params"]
                print(f"Extrinsic z-parameter: {popt[1]:.4f} ± {perr[1]:.4f}")
            
            if "anomalies" in diagnostics and len(diagnostics["anomalies"]) > 0:
                print(f"Potential anomalies detected at indices: {diagnostics['anomalies']}")
            
        elif choice == "5":
            # Run All Analyses
            logger.info("\nRunning comprehensive analysis...")
            
            # Find transition temperature
            logger.info("\n1. Searching for transition temperature...")
            transition_T = search_transition_temperature(data, verbose=False, plot_result=True)
            
            if transition_T is not None:
                logger.info(f"Transition temperature: {transition_T:.1f} K")
                
                # Analyze extrinsic region
                logger.info("\n2. Analyzing extrinsic region (below transition)...")
                extrinsic_data = data[data["Temperature"] <= transition_T]
                process_extrinsic_region(
                    extrinsic_data,
                    [(extrinsic_data["Temperature"].min(), transition_T)],
                    save_plots=save_plots
                )
                
                # Analyze intrinsic region
                logger.info("\n3. Analyzing intrinsic region (above transition)...")
                intrinsic_data = data[data["Temperature"] >= transition_T]
                process_intrinsic_region(
                    intrinsic_data,
                    [(transition_T, intrinsic_data["Temperature"].max())],
                    save_plots=save_plots
                )
            else:
                logger.warning("Could not determine transition temperature. Using diagnostic analysis...")
                
                # Run diagnostics
                logger.info("\nRunning diagnostic analysis...")
                diagnostics = run_diagnostics(data)
                
                # Try default ranges
                logger.info("\nTrying default temperature ranges...")
                T_min, T_max = data["Temperature"].min(), data["Temperature"].max()
                T_mid = (T_min + T_max) / 2
                
                logger.info("\nAnalyzing potential extrinsic region...")
                process_extrinsic_region(
                    data,
                    [(T_min, T_mid)],
                    save_plots=save_plots
                )
                
                logger.info("\nAnalyzing potential intrinsic region...")
                process_intrinsic_region(
                    data,
                    [(T_mid, T_max)],
                    save_plots=save_plots
                )
            
        elif choice == "6":
            # Compare Two Temperature Ranges
            compare_temperature_ranges(data, save_plots=save_plots)
            
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
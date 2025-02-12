#!/usr/bin/env python3
"""
Muon Lifetime Data Analysis Module

This script analyzes muon lifetime data from CSV files by performing exponential decay fits
and providing various analysis modes.

Modes:
    - Basic Fit Analysis: Uses provided (or default) bounds to fit the decay.
    - Search for Optimal Bounds: Finds time bounds that yield a lifetime (τ) closest to a target.
    - Diagnostics: Runs comprehensive diagnostic analyses.
    - All Analyses: Runs the basic fit, optimal-bound search, and diagnostics sequentially.
    - Compare Two Bounds: Compares fits from two user-specified time bounds.

Usage:
    python muon_lifetime.py
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Configure plotting style
plt.style.use('default')


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
            df = pd.read_csv(csv_path, sep=sep, header=None, names=["Time", "Counts"])
            if df["Counts"].notna().sum() > 0:
                logger.info(f"Successfully read CSV using separator '{sep}'")
                return df.astype({"Time": float, "Counts": float})
        except Exception as e:
            logger.debug(f"Failed with separator '{sep}': {e}")
    
    # Try auto-detection as last resort
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python', header=None, names=["Time", "Counts"])
        logger.info("Successfully read CSV by auto-detecting separator")
        return df.astype({"Time": float, "Counts": float})
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")


def truncate_after_zeros(data: pd.DataFrame, threshold: int = 250) -> pd.DataFrame:
    """Truncate data after encountering consecutive zeros."""
    zero_count = 0
    trunc_idx = len(data)

    for i, count in enumerate(data["Counts"]):
        if count == 0:
            zero_count += 1
        else:
            zero_count = 0
        if zero_count >= threshold:
            trunc_idx = max(i - threshold + 10, 0)  # Keep 10 points buffer
            logger.info(f"Truncating at index {trunc_idx} (Time ≈ {data['Time'].iloc[trunc_idx]:.2f})")
            break

    return data.iloc[:trunc_idx]


def exponential_decay(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """Exponential decay model: A * exp(-t/tau) + C"""
    return A * np.exp(-t / tau) + C


def filter_time_range(data: pd.DataFrame, t_min: float, t_max: float) -> pd.DataFrame:
    """Filter data to specified time range."""
    filtered = data[(data["Time"] >= t_min) & (data["Time"] <= t_max)]
    logger.info(f"Filtered to {t_min}–{t_max}. {len(filtered)} points remain.")
    return filtered


def convert_time_units(data: pd.DataFrame, factor: float = 1.0) -> pd.DataFrame:
    """Convert time units by specified factor."""
    df = data.copy()
    df["Time"] *= factor
    logger.info(f"Time values scaled by factor {factor}")
    return df


def fit_exponential(
    t_data: np.ndarray,
    counts: np.ndarray,
    p0: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit exponential decay to data."""
    if p0 is None:
        p0 = [float(np.max(counts)), 2.2, float(np.min(counts))]
    
    popt, pcov = curve_fit(exponential_decay, t_data, counts, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def plot_results(
    t_data: np.ndarray,
    counts: np.ndarray,
    popt: np.ndarray,
    perr: np.ndarray,
    bounds: Tuple[float, float],
    save_fig: bool = False,
    prefix: str = "muon_lifetime"
) -> None:
    """Plot fit results and residuals."""
    # Generate fit curve
    t_fit = np.linspace(float(np.min(t_data)), float(np.max(t_data)), 500)
    counts_fit = exponential_decay(t_fit, *popt)
    residuals = counts - exponential_decay(t_data, *popt)

    # Main fit plot
    plt.figure(figsize=(14, 8))
    plt.scatter(t_data, counts, label='Data', alpha=0.6)
    plt.plot(t_fit, counts_fit, 'r-', label='Fit')
    
    plt.title(f"Muon Lifetime Data {bounds[0]}–{bounds[1]}", fontsize=16)
    plt.xlabel("Time (µs)", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Add fit parameters text box
    fit_text = (
        f"A = {popt[0]:.2f} ± {perr[0]:.2f}\n"
        f"τ = {popt[1]:.2f} ± {perr[1]:.2f} µs\n"
        f"C = {popt[2]:.2f} ± {perr[2]:.2f}"
    )
    plt.text(
        0.80, 0.85, fit_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    if save_fig:
        fit_filename = f"{prefix}_{bounds[0]}_{bounds[1]}_fit.png"
        plt.savefig(fit_filename, bbox_inches="tight")
        logger.info(f"Saved fit plot: {fit_filename}")

    plt.tight_layout()
    plt.show()

    # Residuals plot
    plt.figure(figsize=(14, 4))
    plt.scatter(t_data, residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Residuals", fontsize=16)
    plt.xlabel("Time (µs)", fontsize=14)
    plt.ylabel("Residuals", fontsize=14)
    plt.grid(True)

    if save_fig:
        res_filename = f"{prefix}_{bounds[0]}_{bounds[1]}_residuals.png"
        plt.savefig(res_filename, bbox_inches="tight")
        logger.info(f"Saved residuals plot: {res_filename}")

    plt.tight_layout()
    plt.show()


def process_bounds(
    data: pd.DataFrame,
    bounds_list: List[Tuple[float, float]],
    zero_threshold: int = 250,
    time_factor: float = 1.0,
    save_plots: bool = False
) -> List[Dict[str, Any]]:
    """Process data for multiple time bounds."""
    results = []

    for lb, ub in bounds_list:
        logger.info(f"\nAnalyzing bounds: {lb}–{ub}")
        
        # Filter and clean data
        filtered = filter_time_range(data, lb, ub)
        if filtered.empty:
            logger.warning("No data in range")
            continue

        processed = (convert_time_units(filtered, time_factor)
                     .pipe(truncate_after_zeros, zero_threshold)
                     .query("Counts > 0"))
        
        if processed.empty:
            logger.warning("No valid data after processing")
            continue

        # Fit and plot
        try:
            t_data = processed["Time"].values
            counts = processed["Counts"].values
            popt, perr = fit_exponential(t_data, counts)
            
            logger.info(
                f"Fit results: A={popt[0]:.2f}±{perr[0]:.2f}, "
                f"τ={popt[1]:.2f}±{perr[1]:.2f} µs, "
                f"C={popt[2]:.2f}±{perr[2]:.2f}"
            )

            results.append({
                "bounds": (lb, ub),
                "popt": popt,
                "perr": perr,
                "t_data": t_data,
                "counts": counts
            })

            plot_results(t_data, counts, popt, perr, (lb, ub), save_plots)
            
        except RuntimeError as e:
            logger.error(f"Fit failed: {e}")
            continue

    return results


def search_optimal_bounds(
    data: pd.DataFrame,
    target_tau: float,
    n_lower: int = 20,
    n_upper: int = 20,
    zero_threshold: int = 250,
    time_factor: float = 1.0,
    verbose: bool = True,
    plot_best: bool = True
) -> Optional[Dict[str, Any]]:
    """Search for bounds giving τ closest to target."""
    t_min, t_max = data["Time"].min(), data["Time"].max()
    best_result = None
    min_error = float('inf')
    min_range = 0.1 * (t_max - t_min)

    for lb in np.linspace(t_min, t_max - min_range, n_lower):
        for ub in np.linspace(lb + min_range, t_max, n_upper):
            # Process data
            filtered = (filter_time_range(data, lb, ub)
                        .pipe(convert_time_units, time_factor)
                        .pipe(truncate_after_zeros, zero_threshold)
                        .query("Counts > 0"))
            
            if filtered.empty:
                continue

            # Fit
            try:
                t_data = filtered["Time"].values
                counts = filtered["Counts"].values
                popt, _ = fit_exponential(t_data, counts)
                
                tau_fit = popt[1]
                error = abs(tau_fit - target_tau)
                
                if verbose:
                    logger.info(
                        f"[CANDIDATE] {lb:.2f}–{ub:.2f}: "
                        f"τ={tau_fit:.3f} µs (err={error:.3f} µs)"
                    )

                if error < min_error:
                    min_error = error
                    best_result = {
                        "bounds": (lb, ub),
                        "popt": popt,
                        "tau_fit": tau_fit,
                        "tau_error": error,
                        "t_data": t_data,
                        "counts": counts
                    }
                    
            except RuntimeError as e:
                if verbose:
                    logger.debug(f"Fit failed for {lb:.2f}–{ub:.2f}: {e}")
                continue

    if best_result is None:
        logger.error("No valid candidates found")
        return None

    # Report results
    lb, ub = best_result["bounds"]
    logger.info(
        f"\nBest bounds: {lb:.2f}–{ub:.2f}; "
        f"τ={best_result['tau_fit']:.4f} µs "
        f"(target={target_tau:.4f}, err={best_result['tau_error']:.4f})"
    )

    if plot_best:
        try:
            popt, pcov = curve_fit(
                exponential_decay, 
                best_result["t_data"],
                best_result["counts"],
                p0=best_result["popt"]
            )
            perr = np.sqrt(np.diag(pcov))
            plot_results(
                best_result["t_data"],
                best_result["counts"],
                popt,
                perr,
                best_result["bounds"]
            )
        except RuntimeError as e:
            logger.warning(f"Final plotting fit failed: {e}")

    return best_result


def run_diagnostics(
    data: pd.DataFrame,
    time_factor: float = 1.0
) -> Dict[str, Any]:
    """Run comprehensive diagnostic analysis."""
    logger.info("\nRunning diagnostic analysis...")
    processed = convert_time_units(data, time_factor)
    t_data = processed["Time"].values
    counts = processed["Counts"].values
    diagnostics: Dict[str, Any] = {}

    # Signal quality
    if len(counts) > 100:
        signal = np.mean(counts[:50])
        noise = np.std(counts[-50:])
        snr = signal / noise if noise != 0 else float('inf')
        logger.info(f"Signal-to-noise ratio: {snr:.2f}")
        diagnostics["signal_to_noise"] = snr

    # Baseline drift
    baseline_drift = (
        (np.mean(counts[-20:]) - np.mean(counts[:20])) / 
        (np.mean(counts[:20]) + 1e-9) * 100
    )
    logger.info(f"Baseline drift: {baseline_drift:.1f}%")
    diagnostics["baseline_drift"] = baseline_drift

    # Progressive fits
    diagnostics["progressive_fits"] = {}
    for window in [2.0, 3.0, 4.0, 5.0, 6.0]:
        mask = t_data <= window
        if np.sum(mask) < 10:
            continue
            
        try:
            p0 = [float(np.max(counts[mask])), 2.2, float(np.min(counts[mask]))]
            bounds_fit = ([0, 1.5, 0], [np.inf, 3.0, np.inf])
            popt, pcov = curve_fit(
                exponential_decay,
                t_data[mask],
                counts[mask],
                p0=p0,
                bounds=bounds_fit
            )
            
            tau = popt[1]
            tau_err = math.sqrt(pcov[1, 1])
            accepted_tau = 2.1969811
            pull = (tau - accepted_tau) / tau_err if tau_err != 0 else float('inf')
            
            residuals = counts[mask] - exponential_decay(t_data[mask], *popt)
            dof = np.sum(mask) - 3
            chi2_red = np.sum(residuals**2 / (counts[mask] + 1e-9)) / dof if dof > 0 else float('inf')

            logger.info(
                f"Fit to {window} µs: τ={tau:.4f}±{tau_err:.4f} µs "
                f"(pull={pull:.2f}σ), χ²/DoF={chi2_red:.2f}"
            )
            
            diagnostics["progressive_fits"][window] = {
                "tau": tau,
                "tau_err": tau_err,
                "pull": pull,
                "chi2_reduced": chi2_red,
            }
        except RuntimeError as e:
            logger.warning(f"Fit failed for window {window} µs: {e}")

    # Background analysis
    diagnostics["background"] = {}
    for start, end in [(8, 10), (10, 12), (12, 14)]:
        mask = (t_data >= start) & (t_data < end)
        if np.sum(mask) > 0:
            bg_mean = np.mean(counts[mask])
            bg_std = np.std(counts[mask])
            logger.info(f"Background {start}–{end} µs: {bg_mean:.1f} ± {bg_std:.1f}")
            diagnostics["background"][(start, end)] = (bg_mean, bg_std)

    # Early-time analysis
    early_mask = t_data <= 1.0
    if np.sum(early_mask) > 10:
        try:
            p0 = [float(np.max(counts[early_mask])), 2.2, float(np.min(counts[early_mask]))]
            popt_early, pcov_early = curve_fit(
                exponential_decay, 
                t_data[early_mask], 
                counts[early_mask], 
                p0=p0
            )
            tau_early = popt_early[1]
            tau_early_err = math.sqrt(pcov_early[1, 1])
            logger.info(f"Early-time (≤1 µs) τ: {tau_early:.4f}±{tau_early_err:.4f} µs")
            diagnostics["early_time"] = {
                "tau": tau_early,
                "tau_err": tau_early_err
            }
        except RuntimeError as e:
            logger.warning(f"Early-time fit failed: {e}")

    # Rate dependence
    diagnostics["rate_effects"] = {}
    if len(counts) > 10:
        high_cut = np.percentile(counts, 75)
        low_cut = np.percentile(counts, 25)
        high_mask = counts > high_cut
        low_mask = counts < low_cut

        if np.sum(high_mask) > 10 and np.sum(low_mask) > 10:
            try:
                popt_high, _ = curve_fit(
                    exponential_decay,
                    t_data[high_mask],
                    counts[high_mask],
                    p0=[1, 2.2, 0]
                )
                popt_low, _ = curve_fit(
                    exponential_decay,
                    t_data[low_mask],
                    counts[low_mask],
                    p0=[1, 2.2, 0]
                )
                logger.info(
                    f"Rate dependence: high-rate τ={popt_high[1]:.4f} µs, "
                    f"low-rate τ={popt_low[1]:.4f} µs"
                )
                diagnostics["rate_effects"] = {
                    "high_tau": popt_high[1],
                    "low_tau": popt_low[1]
                }
            except RuntimeError as e:
                logger.warning(f"Rate-dependent fit failed: {e}")

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


def plot_comparative_results(
    results: List[Dict[str, Any]],
    save_fig: bool = False,
    prefix: str = "muon_lifetime"
) -> None:
    """Plot comparative results for two different bounds."""
    if len(results) != 2:
        logger.error("Exactly two results required for comparative plot")
        return

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1])

    # Plot fits
    for idx, result in enumerate(results):
        ax_fit = fig.add_subplot(gs[0, idx])
        
        t_data = result["t_data"]
        counts = result["counts"]
        popt = result["popt"]
        perr = result["perr"]
        bounds = result["bounds"]
        
        # Generate fit curve
        t_fit = np.linspace(float(np.min(t_data)), float(np.max(t_data)), 500)
        counts_fit = exponential_decay(t_fit, *popt)
        
        # Plot data and fit
        ax_fit.scatter(t_data, counts, label='Data', alpha=0.6)
        ax_fit.plot(t_fit, counts_fit, 'r-', label='Fit')
        
        ax_fit.set_title(f"Bounds: {bounds[0]:.2f}–{bounds[1]:.2f} µs", fontsize=14)
        ax_fit.set_xlabel("Time (µs)", fontsize=16)
        ax_fit.set_ylabel("Counts", fontsize=16)
        ax_fit.legend(fontsize=10)
        ax_fit.grid(True)
        
        # Add fit parameters text box
        fit_text = (
            f"A = {popt[0]:.2f} ± {perr[0]:.2f}\n"
            f"τ = {popt[1]:.2f} ± {perr[1]:.2f} µs\n"
            f"C = {popt[2]:.2f} ± {perr[2]:.2f}"
        )
        ax_fit.text(
            0.95, 0.95, fit_text,
            transform=ax_fit.transAxes,
            fontsize=16,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
        )

    # Plot residuals
    for idx, result in enumerate(results):
        ax_res = fig.add_subplot(gs[idx + 1, :])
        
        t_data = result["t_data"]
        counts = result["counts"]
        popt = result["popt"]
        bounds = result["bounds"]
        
        residuals = counts - exponential_decay(t_data, *popt)
        
        ax_res.scatter(t_data, residuals, alpha=0.6)
        ax_res.axhline(0, color='r', linestyle='--')
        
        ax_res.set_title(f"Residuals ({bounds[0]:.2f}–{bounds[1]:.2f} µs)", fontsize=12)
        ax_res.set_xlabel("Time (µs)", fontsize=12)
        ax_res.set_ylabel("Residuals", fontsize=12)
        ax_res.grid(True)

    plt.tight_layout()
    
    if save_fig:
        filename = f"{prefix}_comparative_analysis.png"
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        logger.info(f"Saved comparative plot: {filename}")
    
    plt.show()


def print_menu() -> None:
    """Display the main menu."""
    print("\nMuon Lifetime Analysis Menu")
    print("=" * 30)
    print("1. Basic Fit Analysis")
    print("2. Search for Optimal Bounds")
    print("3. Run Diagnostics")
    print("4. Run All Analyses")
    print("5. Compare Two Bounds")
    print("6. Exit")
    print("=" * 30)


def get_bounds_input() -> Optional[Tuple[float, float]]:
    """Get bounds input from user."""
    try:
        bounds_str = input("Enter bounds as 'lower,upper' (e.g., '0.1,3.5'): ").strip()
        if not bounds_str:
            return None
        lb, ub = map(float, bounds_str.split(','))
        return (lb, ub)
    except ValueError:
        logger.error("Invalid bounds format")
        return None


def compare_two_bounds(data: pd.DataFrame, time_factor: float = 1.0, save_plots: bool = False) -> None:
    """Run comparative analysis for two different bounds."""
    logger.info("\nComparative Bounds Analysis")
    logger.info("Enter bounds for first fit:")
    bounds1 = get_bounds_input()
    if not bounds1:
        return

    logger.info("\nEnter bounds for second fit:")
    bounds2 = get_bounds_input()
    if not bounds2:
        return

    results = process_bounds(
        data,
        [bounds1, bounds2],
        zero_threshold=250,
        time_factor=time_factor,
        save_plots=False  # Don't save individual plots
    )

    if len(results) == 2:
        plot_comparative_results(results, save_fig=save_plots)
    else:
        logger.error("Failed to generate both fits for comparison")


def main() -> None:
    """Interactive main entry point for analysis."""
    while True:
        print_menu()
        choice = input("Enter your choice (1-6): ").strip()
        if choice == "6":
            print("Exiting program...")
            break
        if choice not in ["1", "2", "3", "4", "5"]:
            print("Invalid choice. Please try again.")
            continue

        # Ask which data file to use
        file_path = select_file()
        if not file_path:
            continue

        # Ask user for analysis parameters
        conversion_str = input("Enter time conversion factor (default 1.0): ").strip() or "1.0"
        try:
            conversion = float(conversion_str)
        except ValueError:
            conversion = 1.0
        
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
            # Basic Fit Analysis
            logger.info("\nRunning basic fit analysis...")
            print("\nEnter custom bounds as (lb, ub); (lb2, ub2); etc.")
            print("Example: (0,3.5);(0.1,3.5)")
            custom_bounds_str = input("Press ENTER for default bounds: ").strip()
            default_bounds = [
                (0, 3.5), (0.1, 3.5), (0.3, 3.5),
                (0.5, 3.5), (0, 4.0), (0, 4.5),
                (0, 5), (0, 5.5), (0, 6.0)
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

            results = process_bounds(
                data,
                used_bounds,
                zero_threshold=250,
                time_factor=conversion,
                save_plots=save_plots
            )
            if results:
                logger.info("\nFit Results Summary:")
                for res in results:
                    lb, ub = res["bounds"]
                    popt, perr = res["popt"], res["perr"]
                    logger.info(
                        f"Bounds {lb:.2f}–{ub:.2f}: A={popt[0]:.2f}±{perr[0]:.2f}, "
                        f"τ={popt[1]:.2f}±{perr[1]:.2f} µs, C={popt[2]:.2f}±{perr[2]:.2f}"
                    )
        elif choice == "2":
            # Search for Optimal Bounds
            target_str = input("Enter target τ value (default 2.197 µs): ").strip() or "2.197"
            try:
                target_tau = float(target_str)
            except ValueError:
                target_tau = 2.197
            logger.info(f"\nSearching for optimal bounds (target τ={target_tau} µs)...")
            best_candidate = search_optimal_bounds(
                data,
                target_tau=target_tau,
                zero_threshold=250,
                time_factor=conversion,
                verbose=True,
                plot_best=True
            )
            if best_candidate:
                lb, ub = best_candidate["bounds"]
                logger.info(
                    f"\nOptimal bounds found: {lb:.2f}–{ub:.2f}\n"
                    f"Fitted τ={best_candidate['tau_fit']:.4f} µs\n"
                    f"Deviation from target: {best_candidate['tau_error']:.4f} µs"
                )
        elif choice == "3":
            # Diagnostics
            logger.info("\nRunning diagnostic analysis...")
            diagnostics = run_diagnostics(data, time_factor=conversion)
            logger.info("\nDiagnostic Results Summary:")
            for key, value in diagnostics.items():
                logger.info(f"{key}: {value}")
        elif choice == "4":
            # Run All Analyses: Basic Fit, Search, and Diagnostics
            logger.info("\nRunning basic fit analysis (All Analyses)...")
            default_bounds = [
                (0, 3.5), (0.1, 3.5), (0.3, 3.5),
                (0.5, 3.5), (0, 4.0), (0, 4.5),
                (0, 5), (0, 5.5), (0, 6.0)
            ]
            results = process_bounds(
                data,
                default_bounds,
                zero_threshold=250,
                time_factor=conversion,
                save_plots=save_plots
            )
            if results:
                logger.info("\nFit Results Summary:")
                for res in results:
                    lb, ub = res["bounds"]
                    popt, perr = res["popt"], res["perr"]
                    logger.info(
                        f"Bounds {lb:.2f}–{ub:.2f}: A={popt[0]:.2f}±{perr[0]:.2f}, "
                        f"τ={popt[1]:.2f}±{perr[1]:.2f} µs, C={popt[2]:.2f}±{perr[2]:.2f}"
                    )
            target_str = input("Enter target τ value (default 2.197 µs): ").strip() or "2.197"
            try:
                target_tau = float(target_str)
            except ValueError:
                target_tau = 2.197
            logger.info(f"\nSearching for optimal bounds (target τ={target_tau} µs)...")
            best_candidate = search_optimal_bounds(
                data,
                target_tau=target_tau,
                zero_threshold=250,
                time_factor=conversion,
                verbose=True,
                plot_best=True
            )
            if best_candidate:
                lb, ub = best_candidate["bounds"]
                logger.info(
                    f"\nOptimal bounds found: {lb:.2f}–{ub:.2f}\n"
                    f"Fitted τ={best_candidate['tau_fit']:.4f} µs\n"
                    f"Deviation from target: {best_candidate['tau_error']:.4f} µs"
                )
            logger.info("\nRunning diagnostic analysis (All Analyses)...")
            diagnostics = run_diagnostics(data, time_factor=conversion)
            logger.info("\nDiagnostic Results Summary:")
            for key, value in diagnostics.items():
                logger.info(f"{key}: {value}")
        elif choice == "5":
            # Compare Two Bounds
            compare_two_bounds(data, time_factor=conversion, save_plots=save_plots)
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
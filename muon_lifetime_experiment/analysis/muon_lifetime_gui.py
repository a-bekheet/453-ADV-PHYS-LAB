#!/usr/bin/env python3
"""
Muon Lifetime Analysis GUI Application

This is the GUI version of the muon lifetime analysis tool, providing all the functionality
of the CLI version in a user-friendly interface.
"""

import os
import sys
from typing import Optional, List, Dict, Any, Tuple
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTabWidget, QTextEdit, QGroupBox,
    QMessageBox, QProgressBar, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Configure matplotlib to use Qt6
import matplotlib
matplotlib.use('Qt5Agg')  # This still works with Qt6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import muon_lifetime as ml
import logging
import pandas as pd
import numpy as np

# Configure logging to write to QTextEdit
class QTextEditLogger(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.text_widget.setReadOnly(True)
        
    def emit(self, record):
        msg = self.format(record)
        self.text_widget.append(msg)

class AnalysisWorker(QThread):
    """Worker thread for running analyses without blocking the GUI"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, analysis_type: str, data: pd.DataFrame, params: Dict[str, Any]):
        super().__init__()
        self.analysis_type = analysis_type
        self.data = data.copy()  # Make a copy to avoid thread issues
        self.params = params.copy()
        self._is_running = False
        
    def run(self):
        """Run the analysis in a separate thread"""
        self._is_running = True
        try:
            results = {}
            
            if self.analysis_type == "basic":
                if "bounds" not in self.params:
                    raise ValueError("No bounds specified for basic analysis")
                    
                results = {
                    "basic": ml.process_bounds(
                        self.data,
                        self.params["bounds"],
                        time_factor=self.params.get("time_factor", 1.0),
                        save_plots=self.params.get("save_plots", False)
                    )
                }
                
            elif self.analysis_type == "search":
                if "target_tau" not in self.params:
                    raise ValueError("No target tau specified for search")
                    
                search_result = ml.search_optimal_bounds(
                    self.data,
                    target_tau=self.params["target_tau"],
                    time_factor=self.params.get("time_factor", 1.0),
                    plot_best=True
                )
                if search_result:
                    results = {"search": search_result}
                else:
                    raise ValueError("Search failed to find optimal bounds")
                    
            elif self.analysis_type == "diagnostic":
                results = {
                    "diagnostic": ml.run_diagnostics(
                        self.data,
                        time_factor=self.params.get("time_factor", 1.0)
                    )
                }
            
            if not results:
                raise ValueError(f"No results produced for {self.analysis_type} analysis")
                
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self._is_running = False
    
    def stop(self):
        """Stop the analysis if it's running"""
        self._is_running = False
        self.wait()

class MuonLifetimeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muon Lifetime Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data
        self.current_data: Optional[pd.DataFrame] = None
        self.worker: Optional[AnalysisWorker] = None
        self.bounds_list: List[Tuple[float, float]] = []
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Add tabs
        tabs.addTab(self._create_data_tab(), "Data")
        tabs.addTab(self._create_analysis_tab(), "Analysis")
        tabs.addTab(self._create_plots_tab(), "Plots")
        
        # Add status bar
        self.statusBar().showMessage("Ready")
        
        # Set up logging
        self.log_text = QTextEdit()
        layout.addWidget(self.log_text)
        
        # Configure logging
        log_handler = QTextEditLogger(self.log_text)
        log_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(log_handler)
        logging.getLogger().setLevel(logging.INFO)

    def _create_data_tab(self) -> QWidget:
        """Create the data loading and preview tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File selection
        file_group = QGroupBox("Data File")
        file_layout = QVBoxLayout()
        
        # Add dropdown for data_files directory
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self._handle_file_selection)
        self._update_file_dropdown()
        file_layout.addWidget(self.file_combo)
        
        # Add browse button for custom files
        browse_layout = QHBoxLayout()
        self.file_path_label = QLabel("Or select custom file:")
        browse_layout.addWidget(self.file_path_label)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        browse_layout.addWidget(browse_btn)
        file_layout.addLayout(browse_layout)
        
        # Refresh button for data_files directory
        refresh_btn = QPushButton("Refresh File List")
        refresh_btn.clicked.connect(self._update_file_dropdown)
        file_layout.addWidget(refresh_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        return tab

    def _create_analysis_tab(self) -> QWidget:
        """Create the analysis configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analysis type selection
        analysis_group = QGroupBox("Analysis Configuration")
        analysis_layout = QVBoxLayout()
        
        # Time conversion factor
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Time Conversion Factor:"))
        self.time_factor_spin = QDoubleSpinBox()
        self.time_factor_spin.setValue(1.0)
        self.time_factor_spin.setRange(0.001, 1000.0)
        self.time_factor_spin.setDecimals(3)
        time_layout.addWidget(self.time_factor_spin)
        analysis_layout.addLayout(time_layout)
        
        # Bounds configuration
        bounds_group = QGroupBox("Time Bounds")
        bounds_layout = QVBoxLayout()
        
        # Default bounds
        default_bounds = [(0, 3.5), (0.1, 3.5), (0.3, 3.5),
                         (0.5, 3.5), (0, 4.0), (0, 4.5),
                         (0, 5), (0, 5.5), (0, 6.0)]
        
        # Bounds list widget
        self.bounds_list_widget = QTextEdit()
        self.bounds_list_widget.setPlaceholderText("Each line: lower_bound,upper_bound")
        bounds_layout.addWidget(self.bounds_list_widget)
        
        # Set default bounds button
        default_bounds_btn = QPushButton("Use Default Bounds")
        default_bounds_btn.clicked.connect(lambda: self._set_bounds(default_bounds))
        bounds_layout.addWidget(default_bounds_btn)
        
        # Add single bound controls
        add_bound_layout = QHBoxLayout()
        self.lower_bound_spin = QDoubleSpinBox()
        self.lower_bound_spin.setRange(0, 100)
        self.lower_bound_spin.setDecimals(3)
        self.upper_bound_spin = QDoubleSpinBox()
        self.upper_bound_spin.setRange(0, 100)
        self.upper_bound_spin.setDecimals(3)
        self.upper_bound_spin.setValue(3.5)
        
        add_bound_layout.addWidget(QLabel("Lower:"))
        add_bound_layout.addWidget(self.lower_bound_spin)
        add_bound_layout.addWidget(QLabel("Upper:"))
        add_bound_layout.addWidget(self.upper_bound_spin)
        
        add_bound_btn = QPushButton("Add Bound")
        add_bound_btn.clicked.connect(self._add_single_bound)
        add_bound_layout.addWidget(add_bound_btn)
        
        bounds_layout.addLayout(add_bound_layout)
        bounds_group.setLayout(bounds_layout)
        analysis_layout.addWidget(bounds_group)
        
        # Save plots checkbox
        self.save_plots_check = QCheckBox("Save Plots to Files")
        analysis_layout.addWidget(self.save_plots_check)
        
        # Analysis buttons
        button_layout = QHBoxLayout()
        
        basic_btn = QPushButton("Run Basic Analysis")
        basic_btn.clicked.connect(lambda: self._run_analysis("basic"))
        button_layout.addWidget(basic_btn)
        
        search_btn = QPushButton("Search Optimal Bounds")
        search_btn.clicked.connect(lambda: self._run_analysis("search"))
        button_layout.addWidget(search_btn)
        
        diagnostic_btn = QPushButton("Run Diagnostics")
        diagnostic_btn.clicked.connect(lambda: self._run_analysis("diagnostic"))
        button_layout.addWidget(diagnostic_btn)
        
        analysis_layout.addLayout(button_layout)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return tab

    def _create_plots_tab(self) -> QWidget:
        """Create the plots display tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, tab)
        layout.addWidget(self.toolbar)
        
        # Add clear button
        clear_btn = QPushButton("Clear Plots")
        clear_btn.clicked.connect(self._clear_plots)
        layout.addWidget(clear_btn)
        
        return tab

    def _update_file_dropdown(self):
        """Update the dropdown with files from data_files directory"""
        self.file_combo.clear()
        self.file_combo.addItem("Select a file...")
        
        data_files = ml.get_data_files()
        for file in sorted(data_files):
            self.file_combo.addItem(file)

    def _handle_file_selection(self, index: int):
        """Handle file selection from dropdown"""
        if index <= 0:  # Skip the "Select a file..." item
            return
            
        file_name = self.file_combo.currentText()
        try:
            self.current_data = ml.read_csv_file(file_name)
            self.file_path_label.setText(f"Current file: {file_name}")
            self._update_preview()
            self.statusBar().showMessage(f"Loaded {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def _set_bounds(self, bounds: List[Tuple[float, float]]):
        """Set the bounds in the bounds list widget"""
        bounds_text = "\n".join(f"{lb},{ub}" for lb, ub in bounds)
        self.bounds_list_widget.setText(bounds_text)
        self.bounds_list = bounds

    def _add_single_bound(self):
        """Add a single bound pair to the list"""
        lb = self.lower_bound_spin.value()
        ub = self.upper_bound_spin.value()
        
        if lb >= ub:
            QMessageBox.warning(self, "Invalid Bounds", 
                              "Lower bound must be less than upper bound")
            return
            
        current_text = self.bounds_list_widget.toPlainText()
        new_bound = f"{lb},{ub}"
        if current_text:
            self.bounds_list_widget.setText(f"{current_text}\n{new_bound}")
        else:
            self.bounds_list_widget.setText(new_bound)

    def _parse_bounds(self) -> List[Tuple[float, float]]:
        """Parse bounds from the bounds list widget"""
        bounds = []
        text = self.bounds_list_widget.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, "Warning", "No bounds specified")
            return []
            
        for line in text.split('\n'):
            try:
                lb, ub = map(float, line.strip().split(','))
                if lb >= ub:
                    raise ValueError(f"Invalid bounds: {lb} >= {ub}")
                bounds.append((lb, ub))
            except Exception as e:
                QMessageBox.warning(self, "Invalid Bounds", 
                                  f"Failed to parse bounds: {line}\nError: {str(e)}")
                return []
        
        return bounds

    def _browse_file(self):
        """Open file dialog to select data file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_name:
            try:
                self.current_data = ml.read_csv_file(file_name)
                self.file_path_label.setText(os.path.basename(file_name))
                self._update_preview()
                self.statusBar().showMessage(f"Loaded {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def _update_preview(self):
        """Update the data preview text"""
        if self.current_data is not None:
            preview = (
                f"Data Preview:\n\n"
                f"Shape: {self.current_data.shape}\n\n"
                f"First few rows:\n{self.current_data.head().to_string()}\n\n"
                f"Summary Statistics:\n{self.current_data.describe().to_string()}"
            )
            self.preview_text.setText(preview)

    def _run_analysis(self, analysis_type: str):
        """Run the selected analysis type"""
        try:
            if self.current_data is None:
                QMessageBox.warning(self, "Warning", "Please load a data file first")
                return
            
            # Stop any existing worker
            if self.worker is not None and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait()
            
            # Prepare parameters
            params = {
                "time_factor": self.time_factor_spin.value(),
                "save_plots": self.save_plots_check.isChecked()
            }
            
            if analysis_type == "basic":
                bounds = self._parse_bounds()
                if not bounds:
                    return
                params["bounds"] = bounds
            elif analysis_type == "search":
                target_tau, ok = QInputDialog.getDouble(
                    self, "Target Tau",
                    "Enter target τ value (µs):",
                    2.197, 0.1, 10.0, 4
                )
                if not ok:
                    return
                params["target_tau"] = target_tau
            
            # Create and start worker thread
            self.worker = AnalysisWorker(analysis_type, self.current_data, params)
            self.worker.finished.connect(self._handle_analysis_results)
            self.worker.error.connect(self._handle_analysis_error)
            self.worker.progress.connect(self.progress_bar.setValue)
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage(f"Running {analysis_type} analysis...")
            
            self.worker.start()
            
        except Exception as e:
            self.statusBar().showMessage("Analysis failed")
            QMessageBox.critical(self, "Error", f"Failed to start analysis: {str(e)}")
            logging.error(f"Analysis error: {str(e)}")

    def _handle_analysis_results(self, results: Dict[str, Any]):
        """Handle the analysis results"""
        try:
            self.progress_bar.setVisible(False)
            
            if not results:
                raise ValueError("No results received from analysis")
            
            # Update plots if available
            if "basic" in results or "search" in results:
                self._update_plots(results)
                
            # Log the results
            if "basic" in results:
                for i, result in enumerate(results["basic"]):
                    logging.info(f"\nResult {i+1}:")
                    logging.info(f"Bounds: {result['bounds'][0]:.2f}–{result['bounds'][1]:.2f} µs")
                    logging.info(f"τ = {result['popt'][1]:.3f} ± {result['perr'][1]:.3f} µs")
            
            elif "search" in results:
                result = results["search"]
                logging.info("\nSearch Results:")
                logging.info(f"Optimal bounds: {result['bounds'][0]:.2f}–{result['bounds'][1]:.2f} µs")
                logging.info(f"τ = {result['popt'][1]:.3f} ± {result['perr'][1]:.3f} µs")
            
            elif "diagnostic" in results:
                logging.info("\nDiagnostic Results:")
                for key, value in results["diagnostic"].items():
                    logging.info(f"{key}: {value}")
            
            self.statusBar().showMessage("Analysis completed successfully")
            
        except Exception as e:
            self.statusBar().showMessage("Failed to process results")
            QMessageBox.critical(self, "Error", f"Failed to process results: {str(e)}")
            logging.error(f"Results processing error: {str(e)}")

    def _handle_analysis_error(self, error_msg: str):
        """Handle analysis errors"""
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Analysis failed")
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed: {error_msg}")
        logging.error(f"Analysis error: {error_msg}")

    def _clear_plots(self):
        """Clear all plots"""
        self.figure.clear()
        self.canvas.draw()
        self.statusBar().showMessage("Plots cleared")

    def _update_plots(self, results: Dict[str, Any]):
        """Update the plots tab with new results"""
        try:
            self.figure.clear()
            
            if "basic" in results and results["basic"]:
                # Create subplots for each result
                n_results = len(results["basic"])
                n_rows = (n_results + 1) // 2  # 2 plots per row
                
                for i, result in enumerate(results["basic"]):
                    # Create subplot
                    ax = self.figure.add_subplot(n_rows, 2, i + 1)
                    
                    # Plot data points
                    ax.scatter(result["t_data"], result["counts"], 
                             label="Data", alpha=0.6)
                    
                    # Plot fit line
                    t_fit = np.linspace(min(result["t_data"]), max(result["t_data"]), 500)
                    counts_fit = ml.exponential_decay(t_fit, *result["popt"])
                    ax.plot(t_fit, counts_fit, 'r-', label="Fit")
                    
                    # Add labels and title
                    lb, ub = result["bounds"]
                    ax.set_xlabel("Time (µs)")
                    ax.set_ylabel("Counts")
                    ax.set_title(f"Bounds: {lb:.2f}–{ub:.2f} µs")
                    
                    # Add fit parameters text box
                    fit_text = (
                        f"A = {result['popt'][0]:.2f}±{result['perr'][0]:.2f}\n"
                        f"τ = {result['popt'][1]:.3f}±{result['perr'][1]:.3f} µs\n"
                        f"C = {result['popt'][2]:.2f}±{result['perr'][2]:.2f}"
                    )
                    ax.text(0.95, 0.95, fit_text,
                           transform=ax.transAxes,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    
                    ax.grid(True)
                    ax.legend()

            elif "search" in results and results["search"]:
                # Plot search results
                result = results["search"]
                ax = self.figure.add_subplot(111)
                
                # Plot data points
                ax.scatter(result["t_data"], result["counts"], 
                         label="Data", alpha=0.6)
                
                # Plot fit line
                t_fit = np.linspace(min(result["t_data"]), max(result["t_data"]), 500)
                counts_fit = ml.exponential_decay(t_fit, *result["popt"])
                ax.plot(t_fit, counts_fit, 'r-', label="Best Fit")
                
                # Add labels and title
                lb, ub = result["bounds"]
                ax.set_xlabel("Time (µs)")
                ax.set_ylabel("Counts")
                ax.set_title(f"Optimal Bounds: {lb:.2f}–{ub:.2f} µs")
                
                # Add fit parameters text box
                fit_text = (
                    f"A = {result['popt'][0]:.2f}±{result['perr'][0]:.2f}\n"
                    f"τ = {result['popt'][1]:.3f}±{result['perr'][1]:.3f} µs\n"
                    f"C = {result['popt'][2]:.2f}±{result['perr'][2]:.2f}"
                )
                ax.text(0.95, 0.95, fit_text,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                ax.grid(True)
                ax.legend()
            
            # Adjust layout to prevent overlapping
            self.figure.tight_layout()
            
            # Draw the canvas
            self.canvas.draw_idle()
            
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", 
                              f"Failed to update plots: {str(e)}\n\n"
                              "Please try clearing the plots and running the analysis again.")
            logging.error(f"Plot error: {str(e)}")

    def closeEvent(self, event):
        """Handle application closing"""
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MuonLifetimeGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 
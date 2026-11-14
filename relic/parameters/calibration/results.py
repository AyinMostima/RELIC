"""
Calibration Results and Metrics

This module defines data structures for storing and analyzing parameter
calibration results. It provides comprehensive metrics, visualization
capabilities, and statistical analysis tools for calibration outcomes.

Scientific Background:
    Calibration results require comprehensive documentation and analysis
    to ensure model reliability and reproducibility. Key metrics include:
    - Goodness-of-fit statistics (R^2, Nash-Sutcliffe efficiency)
    - Parameter uncertainty quantification
    - Convergence diagnostics
    - Sensitivity analysis results

References:
    - Nash & Sutcliffe (1970). River flow forecasting through conceptual models.
    - Gupta et al. (1999). Status of automatic calibration for hydrologic models.
    - Bennett et al. (2013). Characterising performance of environmental models.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import numpy as np
import json
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt
from scipy import stats

from ..manager import get_global_manager


class CalibrationStatus(Enum):
    """Status of calibration process"""
    SUCCESS = "success"
    FAILED = "failed"
    TERMINATED_EARLY = "terminated_early"
    TIMEOUT = "timeout"
    USER_INTERRUPTED = "user_interrupted"
    CONVERGENCE_FAILURE = "convergence_failure"
    OBJECTIVE_FAILURE = "objective_failure"


@dataclass
class CalibrationMetrics:
    """Statistical metrics for calibration quality assessment.

    Every statistic is a computed record field without a default; the single
    producer is ``ParameterCalibrator._calculate_metrics``.
    """

    # Goodness-of-fit metrics
    mse: float                          # Mean Squared Error
    rmse: float                         # Root Mean Squared Error
    mae: float                          # Mean Absolute Error
    r_squared: float                    # Coefficient of determination R2 = r^2
    nash_sutcliffe: float               # Nash-Sutcliffe efficiency

    # Bias metrics
    bias: float                         # Mean bias
    percent_bias: float                 # Percent bias (PBIAS)

    # Distribution metrics
    correlation: float                  # Pearson correlation coefficient r

    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationMetrics':
        """Create from dictionary"""
        return cls(**data)

    def get_performance_grade(self, variable_category: str) -> str:
        """
        Rate the calibration with the general performance ratings of Moriasi et al. (2007).

        Parameters:
            variable_category: Category of the calibrated output (streamflow,
                sediment, nitrogen or phosphorus); it selects the percent-bias
                ratings and has no default.

        Returns:
            "Very good", "Good", "Satisfactory" or "Unsatisfactory": the lowest
            of the NSE rating and the PBIAS rating (a simulation is rated at a
            level only when every statistic meets it).
        """
        # References: Moriasi et al. (2007), Trans. ASABE 50(3):885-900, Table 4 (general
        #   performance ratings for a monthly time step) and p. 891.
        # Mathematical expression: rating(NSE) = very good if NSE > n_vg, good if NSE > n_g,
        #   satisfactory if NSE > n_s, else unsatisfactory; rating(PBIAS) = very good if
        #   |PBIAS| < p_vg(c), good if |PBIAS| < p_g(c), satisfactory if |PBIAS| < p_s(c), else
        #   unsatisfactory; grade = min(rating(NSE), rating(PBIAS)).
        # Parameter meanings: n_vg, n_g = validation.calibration.performance_rating.nse_*_min;
        #   n_s = validation.calibration.statistical.min_nash_sutcliffe; p_vg(c), p_g(c) =
        #   validation.calibration.performance_rating.percent_bias_*_max.<c>; p_s(c) =
        #   validation.calibration.statistical.max_abs_percent_bias.<c>; c the variable category.
        # Units: NSE dimensionless; PBIAS and its bounds in percent.
        # Domain/constraints: the ratings are stated for continuous long-term simulation at a
        #   monthly time step; Table 4 has no R2 rating, so R2 is not graded.
        # Sign convention: only |PBIAS| is rated.
        manager = get_global_manager()
        satisfactory_pbias = manager.get_namespace_parameters(
            "validation.calibration.statistical.max_abs_percent_bias"
        )
        categories = sorted(key.rsplit(".", 1)[-1] for key in satisfactory_pbias)
        if variable_category not in categories:
            raise ValueError(
                f"variable_category {variable_category!r} has no declared performance rating; "
                f"declared categories: {categories}"
            )
        labels = ("Unsatisfactory", "Satisfactory", "Good", "Very good")
        nse_bounds = (
            float(manager.get("validation.calibration.statistical.min_nash_sutcliffe", required=True)),
            float(manager.get("validation.calibration.performance_rating.nse_good_min", required=True)),
            float(manager.get("validation.calibration.performance_rating.nse_very_good_min", required=True)),
        )
        pbias_bounds = (
            float(satisfactory_pbias[f"validation.calibration.statistical.max_abs_percent_bias.{variable_category}"]),
            float(manager.get(
                f"validation.calibration.performance_rating.percent_bias_good_max.{variable_category}",
                required=True,
            )),
            float(manager.get(
                f"validation.calibration.performance_rating.percent_bias_very_good_max.{variable_category}",
                required=True,
            )),
        )
        nse_rating = sum(1 for bound in nse_bounds if self.nash_sutcliffe > bound)
        abs_pbias = abs(self.percent_bias)
        pbias_rating = sum(1 for bound in pbias_bounds if abs_pbias < bound)
        return labels[min(nse_rating, pbias_rating)]


@dataclass
class ParameterUncertainty:
    """Parameter uncertainty information"""
    parameter_name: str
    mean: float
    std: float
    confidence_interval_95: Tuple[float, float]
    samples: Optional[np.ndarray] = None
    distribution: Optional[str] = None  # 'normal', 'lognormal', etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        if self.samples is not None:
            data['samples'] = self.samples.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterUncertainty':
        """Create from dictionary"""
        data = data.copy()
        if 'samples' in data and data['samples'] is not None:
            data['samples'] = np.array(data['samples'])
        return cls(**data)


@dataclass
class SensitivityAnalysis:
    """Parameter sensitivity analysis results"""
    parameter_sensitivities: Dict[str, float]  # Parameter -> sensitivity index
    method: str  # Method actually used for the sensitivity analysis (e.g. "sobol", "morris")
    interaction_effects: Optional[Dict[Tuple[str, str], float]] = None
    total_sensitivity_indices: Optional[Dict[str, float]] = None
    first_order_indices: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        if self.interaction_effects:
            # Convert tuple keys to string for JSON serialization
            data['interaction_effects'] = {
                f"{k[0]}_{k[1]}": v for k, v in self.interaction_effects.items()
            }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensitivityAnalysis':
        """Create from dictionary"""
        data = data.copy()
        if 'interaction_effects' in data and data['interaction_effects']:
            # Convert string keys back to tuples
            data['interaction_effects'] = {
                tuple(k.split('_')): v for k, v in data['interaction_effects'].items()
            }
        return cls(**data)


@dataclass
class CalibrationResult:
    """
    Comprehensive calibration result with metrics and analysis
    
    This class stores all information about a parameter calibration run,
    including optimized parameters, convergence history, statistical metrics,
    and uncertainty quantification.
    """
    
    # Basic information
    method: str
    target_parameters: List[str]
    initial_parameters: Dict[str, float]
    optimized_parameters: Dict[str, float]
    parameter_bounds: Dict[str, Any]
    
    # Objective function results
    objective_value: float

    # Convergence information and termination (computed record fields, no defaults)
    iteration_count: int
    elapsed_time: float
    success: bool
    status: CalibrationStatus
    termination_reason: str

    initial_objective_value: Optional[float] = None
    convergence_history: List[float] = field(default_factory=list)

    # Statistical metrics
    metrics: Optional[CalibrationMetrics] = None
    
    # Uncertainty quantification
    parameter_uncertainties: Optional[Dict[str, ParameterUncertainty]] = None
    
    # Sensitivity analysis
    sensitivity_analysis: Optional[SensitivityAnalysis] = None
    
    # Additional data
    model_outputs: Optional[Dict[str, np.ndarray]] = None
    observations: Optional[Dict[str, np.ndarray]] = None
    
    # Metadata
    timestamp: Optional[datetime] = None
    calibrator_config: Optional[Dict[str, Any]] = None
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

        # success and status describe the same outcome; a contradicting pair is an
        # invalid record, not something to reconcile silently.
        if self.success != (self.status is CalibrationStatus.SUCCESS):
            raise ValueError(
                f"CalibrationResult status {self.status.value!r} contradicts success={self.success}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        
        # Handle special fields
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        
        if self.metrics:
            data['metrics'] = self.metrics.to_dict()
        
        if self.parameter_uncertainties:
            data['parameter_uncertainties'] = {
                k: v.to_dict() for k, v in self.parameter_uncertainties.items()
            }
        
        if self.sensitivity_analysis:
            data['sensitivity_analysis'] = self.sensitivity_analysis.to_dict()
        
        # Convert numpy arrays to lists for JSON serialization
        if self.model_outputs:
            data['model_outputs'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.model_outputs.items()
            }
        
        if self.observations:
            data['observations'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.observations.items()
            }
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationResult':
        """Create from dictionary"""
        data = data.copy()
        
        # Handle special fields
        if 'status' in data:
            data['status'] = CalibrationStatus(data['status'])
        
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        if 'metrics' in data and data['metrics']:
            data['metrics'] = CalibrationMetrics.from_dict(data['metrics'])
        
        if 'parameter_uncertainties' in data and data['parameter_uncertainties']:
            data['parameter_uncertainties'] = {
                k: ParameterUncertainty.from_dict(v)
                for k, v in data['parameter_uncertainties'].items()
            }
        
        if 'sensitivity_analysis' in data and data['sensitivity_analysis']:
            data['sensitivity_analysis'] = SensitivityAnalysis.from_dict(data['sensitivity_analysis'])
        
        # Convert lists back to numpy arrays
        if 'model_outputs' in data and data['model_outputs']:
            data['model_outputs'] = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in data['model_outputs'].items()
            }
        
        if 'observations' in data and data['observations']:
            data['observations'] = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in data['observations'].items()
            }
        
        return cls(**data)
    
    def get_parameter_improvement(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate parameter improvement from initial to optimized values
        
        Returns:
            Dictionary with improvement statistics for each parameter
        """
        improvements = {}
        
        for param in self.target_parameters:
            initial = self.initial_parameters.get(param, 0)
            optimized = self.optimized_parameters.get(param, 0)
            
            absolute_change = optimized - initial
            relative_change = (
                absolute_change / initial if initial != 0 else float('inf')
            )
            
            improvements[param] = {
                'initial_value': initial,
                'optimized_value': optimized,
                'absolute_change': absolute_change,
                'relative_change': relative_change,
                'percent_change': relative_change * 100
            }
        
        return improvements
    
    def get_convergence_rate(self) -> Optional[float]:
        """
        Calculate convergence rate from convergence history
        
        Returns:
            Convergence rate (improvement per iteration) or None
        """
        if len(self.convergence_history) < 2:
            return None
        
        # Calculate improvement from first to last
        initial_obj = self.convergence_history[0]
        final_obj = self.convergence_history[-1]
        n_iterations = len(self.convergence_history) - 1
        
        if n_iterations > 0 and initial_obj != final_obj:
            return (initial_obj - final_obj) / n_iterations
        
        return 0.0
    
    def compute_parameter_correlation(self) -> Optional[np.ndarray]:
        """
        Compute parameter correlation matrix from uncertainty samples
        
        Returns:
            Correlation matrix or None if uncertainty data not available
        """
        if not self.parameter_uncertainties:
            return None
        
        # Collect samples for all parameters
        param_names = list(self.parameter_uncertainties.keys())
        samples_list = []
        
        for param in param_names:
            uncertainty = self.parameter_uncertainties[param]
            if uncertainty.samples is not None:
                samples_list.append(uncertainty.samples)
            else:
                return None
        
        if samples_list:
            samples_matrix = np.column_stack(samples_list)
            return np.corrcoef(samples_matrix.T)
        
        return None
    
    def save_to_file(self, file_path: Path) -> None:
        """
        Save calibration result to JSON file

        Parameters:
            file_path: Path to save file

        Raises:
            OSError: If the directory or the file cannot be written
            TypeError, ValueError: If the result is not JSON-serializable
        """
        # The JSON text is built before the file is opened, so a result that
        # cannot be serialized leaves no partial file behind.
        payload = json.dumps(self.to_dict(), indent=2)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(payload)

    @classmethod
    def load_from_file(cls, file_path: Path) -> 'CalibrationResult':
        """
        Load calibration result from JSON file

        Parameters:
            file_path: Path to load from

        Returns:
            CalibrationResult instance

        Raises:
            OSError: If the file cannot be read (FileNotFoundError when absent)
            ValueError: If the file is not valid JSON
            TypeError, KeyError: If the JSON does not describe a CalibrationResult
        """
        with open(file_path, 'r', encoding="utf-8-sig") as f:
            data = json.load(f)

        return cls.from_dict(data)


@dataclass
class CalibrationReport:
    """
    Comprehensive calibration report with visualizations and analysis
    
    This class generates detailed reports about calibration results,
    including statistical analysis, visualization, and recommendations.
    """
    
    result: CalibrationResult
    # Category of the calibrated output variable; it selects the performance
    # ratings of the summary and has no default.
    variable_category: str
    title: str = "Parameter Calibration Report"
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            'calibration_method': self.result.method,
            'target_parameters': self.result.target_parameters,
            'success': self.result.success,
            'status': self.result.status.value,
            'elapsed_time': f"{self.result.elapsed_time:.2f} seconds",
            'iterations': self.result.iteration_count,
            'final_objective': self.result.objective_value,
            'improvement': None,
            'parameter_changes': self.result.get_parameter_improvement(),
            'performance_grade': None
        }
        
        # Calculate objective improvement
        if (self.result.initial_objective_value is not None and
            self.result.initial_objective_value != 0):
            improvement = (
                (self.result.initial_objective_value - self.result.objective_value) /
                self.result.initial_objective_value * 100
            )
            summary['improvement'] = f"{improvement:.2f}%"
        
        # Performance grade
        if self.result.metrics:
            summary['performance_grade'] = self.result.metrics.get_performance_grade(self.variable_category)
            summary['nash_sutcliffe'] = self.result.metrics.nash_sutcliffe
            summary['r_squared'] = self.result.metrics.r_squared
            summary['rmse'] = self.result.metrics.rmse
        
        return summary
    
    def create_convergence_plot(self, save_path: Optional[Path] = None) -> None:
        """Create convergence plot"""
        if not self.result.convergence_history:
            return
        
        plt.figure(figsize=(10, 6))
        
        iterations = range(len(self.result.convergence_history))
        plt.plot(iterations, self.result.convergence_history, 'b-', linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value')
        plt.title('Calibration Convergence History')
        plt.grid(True, alpha=0.3)
        
        # Add final value annotation
        final_value = self.result.convergence_history[-1]
        plt.annotate(
            f'Final: {final_value:.6e}',
            xy=(len(iterations)-1, final_value),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_parameter_comparison_plot(self, save_path: Optional[Path] = None) -> None:
        """Create parameter before/after comparison plot"""
        improvements = self.result.get_parameter_improvement()
        
        if not improvements:
            return
        
        params = list(improvements.keys())
        initial_values = [improvements[p]['initial_value'] for p in params]
        optimized_values = [improvements[p]['optimized_value'] for p in params]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot comparison
        x = np.arange(len(params))
        width = 0.35
        
        ax1.bar(x - width/2, initial_values, width, label='Initial', alpha=0.7)
        ax1.bar(x + width/2, optimized_values, width, label='Optimized', alpha=0.7)
        
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Parameter Values')
        ax1.set_title('Parameter Values: Initial vs Optimized')
        ax1.set_xticks(x)
        ax1.set_xticklabels(params, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Relative change plot
        relative_changes = [improvements[p]['percent_change'] for p in params]
        colors = ['red' if x < 0 else 'green' for x in relative_changes]
        
        ax2.bar(x, relative_changes, color=colors, alpha=0.7)
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Percent Change (%)')
        ax2.set_title('Parameter Changes (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(params, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_goodness_of_fit_plot(self, save_path: Optional[Path] = None) -> None:
        """Create goodness-of-fit scatter plot"""
        if not (self.result.model_outputs and self.result.observations):
            return
        
        # Combine all model outputs and observations
        model_values = []
        obs_values = []
        
        for var_name in self.result.observations.keys():
            if var_name in self.result.model_outputs:
                obs_values.extend(self.result.observations[var_name].flatten())
                model_values.extend(self.result.model_outputs[var_name].flatten())
        
        if not model_values:
            return
        
        model_values = np.array(model_values)
        obs_values = np.array(obs_values)
        
        plt.figure(figsize=(8, 8))
        
        # Scatter plot
        plt.scatter(obs_values, model_values, alpha=0.6, s=20)
        
        # 1:1 line
        min_val = min(np.min(obs_values), np.min(model_values))
        max_val = max(np.max(obs_values), np.max(model_values))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
        
        # Best fit line
        slope, intercept, r_value, p_value, std_err = stats.linregress(obs_values, model_values)
        line_x = np.array([min_val, max_val])
        line_y = slope * line_x + intercept
        plt.plot(line_x, line_y, 'b-', linewidth=2, 
                label=f'Best Fit (R^2 = {r_value**2:.3f})')
        
        plt.xlabel('Observed Values')
        plt.ylabel('Modeled Values')
        plt.title('Goodness of Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        if self.result.metrics:
            stats_text = (
                f"RMSE: {self.result.metrics.rmse:.4f}\n"
                f"Nash-Sutcliffe: {self.result.metrics.nash_sutcliffe:.3f}\n"
                f"Bias: {self.result.metrics.bias:.4f}"
            )
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_sensitivity_plot(self, save_path: Optional[Path] = None) -> None:
        """Create parameter sensitivity plot"""
        if not self.result.sensitivity_analysis:
            return
        
        sensitivities = self.result.sensitivity_analysis.parameter_sensitivities
        
        if not sensitivities:
            return
        
        params = list(sensitivities.keys())
        values = list(sensitivities.values())
        
        plt.figure(figsize=(10, 6))
        
        # Sort by sensitivity value
        sorted_indices = np.argsort(values)[::-1]
        sorted_params = [params[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_params)))
        bars = plt.bar(range(len(sorted_params)), sorted_values, color=colors)
        
        plt.xlabel('Parameters')
        plt.ylabel('Sensitivity Index')
        plt.title('Parameter Sensitivity Analysis')
        plt.xticks(range(len(sorted_params)), sorted_params, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sorted_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def generate_full_report(self, output_dir: Path) -> Path:
        """
        Generate comprehensive calibration report
        
        Parameters:
            output_dir: Directory to save report files
            
        Returns:
            Path to main report file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        self.create_convergence_plot(output_dir / "convergence.png")
        self.create_parameter_comparison_plot(output_dir / "parameter_comparison.png")
        self.create_goodness_of_fit_plot(output_dir / "goodness_of_fit.png")
        self.create_sensitivity_plot(output_dir / "sensitivity.png")
        
        # Generate summary
        summary = self.generate_summary()
        
        # Create HTML report
        html_content = self._generate_html_report(summary, output_dir)
        report_file = output_dir / "calibration_report.html"
        
        with open(report_file, 'w', encoding="utf-8") as f:
            f.write(html_content)
        
        # Save raw results
        self.result.save_to_file(output_dir / "calibration_results.json")
        
        return report_file
    
    def _generate_html_report(self, summary: Dict[str, Any], output_dir: Path) -> str:
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #333; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 5px 0; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .image {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{self.title}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Calibration Summary</h2>
                <div class="metric">Method: {summary['calibration_method']}</div>
                <div class="metric">Status: <span class="{'success' if summary['success'] else 'failure'}">{summary['status']}</span></div>
                <div class="metric">Elapsed Time: {summary['elapsed_time']}</div>
                <div class="metric">Iterations: {summary['iterations']}</div>
                <div class="metric">Final Objective: {summary['final_objective']:.6e}</div>
                {f"<div class='metric'>Improvement: {summary['improvement']}</div>" if summary['improvement'] else ""}
                {f"<div class='metric'>Performance Grade: {summary['performance_grade']}</div>" if summary['performance_grade'] else ""}
            </div>
            
            <div class="section">
                <h2>Target Parameters</h2>
                <ul>
                    {"".join([f"<li>{param}</li>" for param in summary['target_parameters']])}
                </ul>
            </div>
            
            <div class="section">
                <h2>Statistical Metrics</h2>
                {f"<div class='metric'>Nash-Sutcliffe Efficiency: {summary.get('nash_sutcliffe', 'N/A'):.3f}</div>" if 'nash_sutcliffe' in summary else ""}
                {f"<div class='metric'>R^2: {summary.get('r_squared', 'N/A'):.3f}</div>" if 'r_squared' in summary else ""}
                {f"<div class='metric'>RMSE: {summary.get('rmse', 'N/A'):.6f}</div>" if 'rmse' in summary else ""}
            </div>
            
            <div class="section">
                <h2>Parameter Changes</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Initial Value</th>
                        <th>Optimized Value</th>
                        <th>Absolute Change</th>
                        <th>Percent Change</th>
                    </tr>
                    {"".join([
                        f"<tr><td>{param}</td><td>{changes['initial_value']:.6f}</td><td>{changes['optimized_value']:.6f}</td><td>{changes['absolute_change']:.6f}</td><td>{changes['percent_change']:.2f}%</td></tr>"
                        for param, changes in summary['parameter_changes'].items()
                    ])}
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="image">
                    <h3>Convergence History</h3>
                    <img src="convergence.png" alt="Convergence Plot" style="max-width: 100%;">
                </div>
                <div class="image">
                    <h3>Parameter Comparison</h3>
                    <img src="parameter_comparison.png" alt="Parameter Comparison" style="max-width: 100%;">
                </div>
                <div class="image">
                    <h3>Goodness of Fit</h3>
                    <img src="goodness_of_fit.png" alt="Goodness of Fit" style="max-width: 100%;">
                </div>
                <div class="image">
                    <h3>Parameter Sensitivity</h3>
                    <img src="sensitivity.png" alt="Sensitivity Analysis" style="max-width: 100%;">
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
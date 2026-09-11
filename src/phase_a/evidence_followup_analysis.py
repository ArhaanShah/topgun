"""Analysis pipeline for evidence followup experiment.

Handles factor-aware contrasts, window-specific labels, and Bayesian inference.
CPU-only: no GPU dependencies or model imports required.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class AnalysisConfig:
    """Analysis settings from frozen design."""
    posterior_prior_alpha: float = 0.5
    posterior_prior_beta: float = 0.5
    posterior_draws: int = 20000
    sensitivity_prior_alpha: float = 1.0
    sensitivity_prior_beta: float = 1.0
    credible_interval: float = 0.95


class BayesianPosterior:
    """Compute Bayesian posteriors for binary labels with Beta prior."""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, draws: int = 20000):
        self.alpha = alpha
        self.beta = beta
        self.draws = draws
        self.rng = np.random.RandomState(20260914)  # Analysis seed
    
    def posterior_interval(self, positives: int, total: int, credible: float = 0.95) -> dict[str, float]:
        """Compute credible interval for posterior probability."""
        if total == 0:
            return {
                "rate": None,
                "mean": None,
                "median": None,
                "lower": None,
                "upper": None,
                "credible_level": credible,
            }
        
        # Posterior is Beta(alpha + positives, beta + negatives)
        alpha_post = self.alpha + positives
        beta_post = self.beta + (total - positives)
        
        # Draw samples
        samples = self.rng.beta(alpha_post, beta_post, self.draws)
        
        # Compute statistics
        mean = np.mean(samples)
        median = np.median(samples)
        lower, upper = np.percentile(samples, [(1 - credible) / 2 * 100, (1 + credible) / 2 * 100])
        
        return {
            "rate": positives / total,
            "mean": float(mean),
            "median": float(median),
            "lower": float(lower),
            "upper": float(upper),
            "credible_level": credible,
            "n": total,
        }
    
    def contrast_posterior(self, pos_a: int, total_a: int, pos_b: int, 
                          total_b: int, credible: float = 0.95) -> dict[str, float]:
        """Compute credible interval for difference in rates (B - A)."""
        if total_a == 0 or total_b == 0:
            return {
                "difference": None,
                "mean_difference": None,
                "lower": None,
                "upper": None,
            }
        
        alpha_a = self.alpha + pos_a
        beta_a = self.beta + (total_a - pos_a)
        alpha_b = self.alpha + pos_b
        beta_b = self.beta + (total_b - pos_b)
        
        # Draw samples and compute difference
        samples_a = self.rng.beta(alpha_a, beta_a, self.draws)
        samples_b = self.rng.beta(alpha_b, beta_b, self.draws)
        diff_samples = samples_b - samples_a
        
        mean_diff = np.mean(diff_samples)
        lower, upper = np.percentile(diff_samples, [(1 - credible) / 2 * 100, (1 + credible) / 2 * 100])
        
        return {
            "difference": (pos_b / total_b) - (pos_a / total_a) if total_a and total_b else None,
            "mean_difference": float(mean_diff),
            "lower": float(lower),
            "upper": float(upper),
            "credible_level": credible,
        }


class ExperimentAnalyzer:
    """Analyze evidence followup results across experiments and factors."""
    
    def __init__(self, responses_path: Path, labels_path: Path, config: AnalysisConfig | None = None):
        self.responses_path = responses_path
        self.labels_path = labels_path
        self.config = config or AnalysisConfig()
        self.posterior = BayesianPosterior(
            alpha=self.config.posterior_prior_alpha,
            beta=self.config.posterior_prior_beta,
            draws=self.config.posterior_draws,
        )
        self.responses: dict[str, dict[str, Any]] = {}
        self.labels: dict[str, dict[str, Any]] = {}
    
    def load_responses(self) -> None:
        """Load all response records."""
        if not self.responses_path.exists():
            return
        with open(self.responses_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                response_id = row.get("response_id")
                if response_id:
                    self.responses[response_id] = row
    
    def load_labels(self) -> None:
        """Load human-labeled responses."""
        if not self.labels_path.exists():
            return
        with open(self.labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                response_id = row.get("response_id")
                if response_id:
                    self.labels[response_id] = row
    
    def get_labeled_subset(self, experiment: str | None = None, 
                          window: str = "full") -> dict[str, int]:
        """Get positive/total counts for a subset of responses."""
        positives = 0
        total = 0
        
        for response_id, label_row in self.labels.items():
            if experiment and label_row.get("experiment") != experiment:
                continue
            
            # Get label for this window
            label_key = f"label_{window}"
            if label_key not in label_row:
                continue
            
            label_val = label_row[label_key]
            if label_val == "":
                continue
            
            total += 1
            if label_val in {"true", "True", "1"}:
                positives += 1
        
        return {"positives": positives, "total": total}
    
    def analyze_experiment_effects(self, window: str = "full") -> dict[str, Any]:
        """Analyze primary effects within each experiment."""
        results = {}
        
        # Experiment A: Evidence effect (within natural answers)
        a_e0 = self.get_labeled_subset(experiment="A")  # A already filtered by labels
        # TODO: need to refilter by evidence level in labels
        results["A_evidence_effect"] = {"status": "needs_implementation"}
        
        # Experiment B: Order effect and evidence interaction
        results["B_order_effect"] = {"status": "needs_implementation"}
        results["B_evidence_effect"] = {"status": "needs_implementation"}
        
        # Experiment C: Cue effect
        results["C_cue_effect"] = {"status": "needs_implementation"}
        
        return results
    
    def analyze_task_transfer(self, window: str = "full") -> dict[str, Any]:
        """Analyze whether effects replicate across tasks."""
        results = {}
        
        for task in ["S", "D", "Z"]:
            # Compute contrasts within each task
            results[f"task_{task}"] = {"status": "needs_implementation"}
        
        return results
    
    def generate_report(self, output_path: Path) -> None:
        """Generate analysis report."""
        self.load_responses()
        self.load_labels()
        
        report = {
            "timestamp": str(Path.cwd()),
            "responses_loaded": len(self.responses),
            "labels_loaded": len(self.labels),
            "experiment_effects": self.analyze_experiment_effects(),
            "task_transfer": self.analyze_task_transfer(),
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)


def main() -> None:
    """CLI for analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze evidence followup responses")
    parser.add_argument("--responses", type=Path, required=True, help="Responses CSV")
    parser.add_argument("--labels", type=Path, required=True, help="Labels CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output report JSON")
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer(args.responses, args.labels)
    analyzer.generate_report(args.output)


if __name__ == "__main__":
    main()

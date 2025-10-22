"""
Cross-dataset evaluation utilities
Tests model generalization to unseen datasets beyond ARAD-1K
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm
from .spectral_utils import calculate_rmse, calculate_psnr, calculate_sam, calculate_mrae


class CrossDatasetEvaluator:
    """
    Evaluates model performance across multiple datasets
    Measures generalization capability
    """
    def __init__(
        self,
        model,
        device='cuda',
        metrics=('rmse', 'psnr', 'sam', 'mrae')
    ):
        """
        Args:
            model: HSI reconstruction model
            device: Device for computation
            metrics: Metrics to compute
        """
        self.model = model
        self.device = device
        self.metrics = metrics
        self.results = {}

    def evaluate_dataset(
        self,
        dataset_name: str,
        data_loader,
        num_samples: Optional[int] = None,
        save_predictions: bool = False,
        output_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a specific dataset

        Args:
            dataset_name: Name of the dataset
            data_loader: DataLoader for the dataset
            num_samples: Number of samples to evaluate (None = all)
            save_predictions: Whether to save predictions
            output_dir: Directory to save predictions

        Returns:
            Dictionary of metric results
        """
        self.model.eval()

        # Initialize metric accumulators
        metric_sums = {metric: 0.0 for metric in self.metrics}
        count = 0

        predictions = []
        targets = []
        filenames = []

        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"Evaluating {dataset_name}")

            for batch_idx, batch in enumerate(pbar):
                # Check if we've reached the sample limit
                if num_samples is not None and count >= num_samples:
                    break

                # Move data to device
                rgb_imgs = batch['rgb'].to(self.device)
                hsi_data = batch['hsi'].to(self.device)

                # Run inference (without masking)
                outputs = self.model(rgb_imgs, use_masking=False)
                hsi_pred = outputs['hsi_output']

                # Compute metrics
                batch_metrics = self._compute_metrics(hsi_pred, hsi_data)

                # Accumulate
                for metric, value in batch_metrics.items():
                    metric_sums[metric] += value * rgb_imgs.shape[0]

                count += rgb_imgs.shape[0]

                # Save predictions if requested
                if save_predictions:
                    predictions.extend(hsi_pred.cpu().numpy())
                    targets.extend(hsi_data.cpu().numpy())
                    if 'filename' in batch:
                        filenames.extend(batch['filename'])

                # Update progress
                avg_metrics = {k: v / count for k, v in metric_sums.items()}
                pbar.set_postfix(avg_metrics)

        # Compute average metrics
        avg_metrics = {metric: metric_sums[metric] / count for metric in self.metrics}

        # Save results
        self.results[dataset_name] = avg_metrics

        # Save predictions if requested
        if save_predictions and output_dir is not None:
            self._save_predictions(
                dataset_name,
                predictions,
                targets,
                filenames,
                output_dir
            )

        return avg_metrics

    def _compute_metrics(
        self,
        pred_hsi: torch.Tensor,
        target_hsi: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all metrics for a batch"""
        metrics = {}

        if 'rmse' in self.metrics:
            metrics['rmse'] = calculate_rmse(pred_hsi, target_hsi).item()

        if 'psnr' in self.metrics:
            metrics['psnr'] = calculate_psnr(pred_hsi, target_hsi).item()

        if 'sam' in self.metrics:
            metrics['sam'] = calculate_sam(pred_hsi, target_hsi).item()

        if 'mrae' in self.metrics:
            metrics['mrae'] = calculate_mrae(pred_hsi, target_hsi).item()

        return metrics

    def _save_predictions(
        self,
        dataset_name: str,
        predictions: List,
        targets: List,
        filenames: List,
        output_dir: Path
    ):
        """Save predictions for analysis"""
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for i, (pred, target, filename) in enumerate(zip(predictions, targets, filenames)):
            save_path = dataset_dir / f"{filename}_pred.npy"
            np.save(save_path, pred)

            if i == 0:  # Save one target as reference
                np.save(dataset_dir / f"{filename}_target.npy", target)

    def compare_datasets(self) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across all evaluated datasets

        Returns:
            Dictionary with dataset comparisons
        """
        if not self.results:
            return {}

        # Compute statistics across datasets
        comparison = {}

        for metric in self.metrics:
            metric_values = [
                results[metric]
                for results in self.results.values()
            ]

            comparison[metric] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'min': np.min(metric_values),
                'max': np.max(metric_values)
            }

        return comparison

    def save_results(self, output_path: Path):
        """Save evaluation results to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            'per_dataset': self.results,
            'comparison': self.compare_datasets()
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {output_path}")

    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("CROSS-DATASET EVALUATION SUMMARY")
        print("="*60)

        for dataset_name, metrics in self.results.items():
            print(f"\n{dataset_name}:")
            for metric, value in metrics.items():
                print(f"  {metric.upper()}: {value:.6f}")

        print("\n" + "-"*60)
        print("OVERALL STATISTICS:")
        print("-"*60)

        comparison = self.compare_datasets()
        for metric, stats in comparison.items():
            print(f"\n{metric.upper()}:")
            print(f"  Mean ± Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")

        print("\n" + "="*60)


class DomainShiftAnalyzer:
    """
    Analyzes domain shift between datasets
    Helps understand generalization challenges
    """
    def __init__(self):
        self.dataset_stats = {}

    def analyze_dataset(
        self,
        dataset_name: str,
        data_loader,
        num_samples: int = 100
    ):
        """
        Analyze statistical properties of a dataset

        Args:
            dataset_name: Name of the dataset
            data_loader: DataLoader for the dataset
            num_samples: Number of samples to analyze
        """
        rgb_pixels = []
        hsi_pixels = []

        count = 0
        for batch in tqdm(data_loader, desc=f"Analyzing {dataset_name}"):
            if count >= num_samples:
                break

            rgb = batch['rgb'].cpu().numpy()
            hsi = batch['hsi'].cpu().numpy()

            # Collect pixel statistics
            rgb_pixels.append(rgb.reshape(-1, rgb.shape[1]))
            hsi_pixels.append(hsi.reshape(-1, hsi.shape[1]))

            count += rgb.shape[0]

        # Concatenate all pixels
        rgb_pixels = np.concatenate(rgb_pixels, axis=0)
        hsi_pixels = np.concatenate(hsi_pixels, axis=0)

        # Compute statistics
        stats = {
            'rgb': {
                'mean': rgb_pixels.mean(axis=0).tolist(),
                'std': rgb_pixels.std(axis=0).tolist(),
                'min': rgb_pixels.min(axis=0).tolist(),
                'max': rgb_pixels.max(axis=0).tolist()
            },
            'hsi': {
                'mean': hsi_pixels.mean(axis=0).tolist(),
                'std': hsi_pixels.std(axis=0).tolist(),
                'min': hsi_pixels.min(axis=0).tolist(),
                'max': hsi_pixels.max(axis=0).tolist()
            }
        }

        self.dataset_stats[dataset_name] = stats

    def compute_domain_distance(
        self,
        dataset1: str,
        dataset2: str,
        modality: str = 'rgb'
    ) -> float:
        """
        Compute domain distance between two datasets

        Args:
            dataset1: First dataset name
            dataset2: Second dataset name
            modality: 'rgb' or 'hsi'

        Returns:
            Domain distance (L2 distance between mean statistics)
        """
        if dataset1 not in self.dataset_stats or dataset2 not in self.dataset_stats:
            return None

        stats1 = self.dataset_stats[dataset1][modality]
        stats2 = self.dataset_stats[dataset2][modality]

        # Compute L2 distance between means
        mean1 = np.array(stats1['mean'])
        mean2 = np.array(stats2['mean'])

        distance = np.linalg.norm(mean1 - mean2)

        return distance

    def save_analysis(self, output_path: Path):
        """Save domain shift analysis"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.dataset_stats, f, indent=2)

        print(f"Domain shift analysis saved to {output_path}")


class GeneralizationMetrics:
    """
    Computes generalization-specific metrics
    """
    @staticmethod
    def compute_consistency_score(
        predictions: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute consistency of predictions across datasets
        Lower is better (more consistent)

        Args:
            predictions: Dict mapping dataset names to prediction arrays

        Returns:
            Consistency score (variance across datasets)
        """
        # Compute variance of metrics across datasets
        all_preds = list(predictions.values())
        stacked = np.stack(all_preds, axis=0)  # [num_datasets, ...]

        # Compute variance along dataset dimension
        variance = np.var(stacked, axis=0).mean()

        return variance

    @staticmethod
    def compute_robustness_score(
        source_metrics: Dict[str, float],
        target_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute robustness scores (performance degradation)

        Args:
            source_metrics: Metrics on source (training) dataset
            target_metrics: Metrics on target (test) dataset

        Returns:
            Dictionary of degradation percentages
        """
        degradation = {}

        for metric in source_metrics:
            if metric not in target_metrics:
                continue

            source_val = source_metrics[metric]
            target_val = target_metrics[metric]

            # For PSNR (higher is better)
            if metric == 'psnr':
                deg = (source_val - target_val) / source_val * 100
            # For RMSE, SAM, MRAE (lower is better)
            else:
                deg = (target_val - source_val) / source_val * 100

            degradation[f'{metric}_degradation_%'] = deg

        return degradation


class TransferLearningEvaluator:
    """
    Evaluates transfer learning performance
    Tests few-shot adaptation to new datasets
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def evaluate_few_shot(
        self,
        support_loader,
        query_loader,
        num_adaptation_steps: int = 10,
        adaptation_lr: float = 1e-5
    ) -> Dict[str, float]:
        """
        Evaluate few-shot adaptation

        Args:
            support_loader: Support set for adaptation
            query_loader: Query set for evaluation
            num_adaptation_steps: Number of adaptation steps
            adaptation_lr: Learning rate for adaptation

        Returns:
            Metrics after adaptation
        """
        # Clone model for adaptation
        adapted_model = type(self.model)(
            **{k: v for k, v in self.model.__dict__.items() if not k.startswith('_')}
        ).to(self.device)
        adapted_model.load_state_dict(self.model.state_dict())

        # Adapt on support set
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=adaptation_lr)

        adapted_model.train()
        for step in range(num_adaptation_steps):
            for batch in support_loader:
                rgb = batch['rgb'].to(self.device)
                hsi = batch['hsi'].to(self.device)

                # Forward pass
                outputs = adapted_model(rgb, use_masking=False)

                # Compute loss
                losses = adapted_model.calculate_losses(outputs, rgb, hsi)
                loss = losses['l1_loss'] + losses['cycle_loss']

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on query set
        adapted_model.eval()
        evaluator = CrossDatasetEvaluator(adapted_model, self.device)
        results = evaluator.evaluate_dataset('few_shot_query', query_loader)

        return results

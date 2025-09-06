import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import heapq


class L1InftyInftyNorm:
    """Implementation of L1,∞,∞ norm based on the provided C++ code"""

    @staticmethod
    def norm(y: np.ndarray) -> float:
        """
        Compute L1,∞,∞ norm: sum over first dimension of max over other dimensions
        """
        if len(y.shape) == 2:
            # For 2D: treat as (d1, d2) where d3=1
            return np.sum(np.max(np.abs(y), axis=1))
        elif len(y.shape) == 3:
            # For 3D: proper L1,∞,∞ norm
            d1, d2, d3 = y.shape
            s = 0
            for i in range(d1):
                max_v = np.max(np.abs(y[i, :, :]))
                s += max_v
            return s
        else:
            # Flatten to 2D and compute
            y_2d = y.reshape(y.shape[0], -1)
            return np.sum(np.max(np.abs(y_2d), axis=1))


class APBOptimizer:
    """Automatic Prune Binarization (APB) algorithm implementation"""

    def __init__(self, binary_threshold: float = 0.1, fp_ratio: float = 0.05):
        self.binary_threshold = binary_threshold
        self.fp_ratio = fp_ratio  # Ratio of weights to keep in full precision

    def compute_importance_score(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute importance score for each weight"""
        # Use gradient-based importance (simplified version)
        # In practice, this would use gradients from backpropagation
        weight_abs = torch.abs(weight)
        weight_var = torch.var(weight_abs)
        weight_mean = torch.mean(weight_abs)

        # Importance score combines magnitude and variance
        importance = weight_abs * (weight_var + weight_mean)
        return importance

    def select_full_precision_weights(self, weight: torch.Tensor,
                                      importance: torch.Tensor) -> torch.Tensor:
        """Select which weights to keep in full precision"""
        flat_importance = importance.flatten()
        num_fp_weights = int(self.fp_ratio * flat_importance.numel())

        if num_fp_weights == 0:
            return torch.zeros_like(weight, dtype=torch.bool)

        # Get top-k important weights
        _, top_indices = torch.topk(flat_importance, num_fp_weights)

        fp_mask = torch.zeros_like(flat_importance, dtype=torch.bool)
        fp_mask[top_indices] = True

        return fp_mask.reshape(weight.shape)

    def binarize_weights(self, weight: torch.Tensor, fp_mask: torch.Tensor) -> torch.Tensor:
        """Binarize weights except those marked for full precision"""
        binary_weight = weight.clone()

        # Binary part: sign of weights not in full precision
        binary_part = torch.sign(weight) * (1 - fp_mask.float())

        # Full precision part: original weights for selected positions
        fp_part = weight * fp_mask.float()

        return binary_part + fp_part


class LayerDistanceCalculator:
    """Calculate distances between layers for filtering"""

    def __init__(self, distance_threshold: float = 0.95):
        self.distance_threshold = distance_threshold

    def compute_layer_features(self, weight: torch.Tensor) -> np.ndarray:
        """Extract features from a layer for distance calculation"""
        weight_np = weight.detach().cpu().numpy()

        # Flatten weight and compute statistical features
        flat_weight = weight_np.flatten()

        features = [
            np.mean(np.abs(flat_weight)),  # Mean absolute value
            np.std(flat_weight),  # Standard deviation
            np.percentile(np.abs(flat_weight), 95),  # 95th percentile
            L1InftyInftyNorm.norm(weight_np),  # L1,∞,∞ norm
            np.max(np.abs(flat_weight)),  # Max absolute value
            np.count_nonzero(flat_weight) / len(flat_weight)  # Density
        ]

        return np.array(features)

    def compute_similarity_matrix(self, layer_features: List[np.ndarray]) -> np.ndarray:
        """Compute cosine similarity matrix between layers"""
        feature_matrix = np.stack(layer_features)
        return cosine_similarity(feature_matrix)

    def find_similar_layers_lis_style(self, similarities: np.ndarray) -> List[int]:
        """
        Find layers to keep using LIS-style approach
        Keep layers that form an increasing sequence of dissimilarity
        """
        n = len(similarities)
        if n <= 1:
            return list(range(n))

        # Convert similarity to dissimilarity
        dissimilarities = 1 - similarities

        # Find longest increasing subsequence of dissimilar layers
        dp = [1] * n  # dp[i] = length of LIS ending at i
        parent = [-1] * n

        for i in range(1, n):
            for j in range(i):
                # If layer i is sufficiently different from layer j
                avg_dissim = np.mean(dissimilarities[i, :j + 1])
                if avg_dissim > (1 - self.distance_threshold) and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        # Reconstruct the sequence
        max_length_idx = np.argmax(dp)
        selected_layers = []
        current = max_length_idx

        while current != -1:
            selected_layers.append(current)
            current = parent[current]

        return selected_layers[::-1]  # Reverse to get correct order

    def filter_similar_layers(self, layer_weights: List[torch.Tensor]) -> List[int]:
        """Filter out similar layers, keeping only diverse ones"""
        if len(layer_weights) <= 1:
            return list(range(len(layer_weights)))

        # Compute features for each layer
        layer_features = [self.compute_layer_features(w) for w in layer_weights]

        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(layer_features)

        # Find layers to keep using LIS-style approach
        selected_indices = self.find_similar_layers_lis_style(similarity_matrix)

        return selected_indices


class AdvancedL1Pruner:
    """Advanced pruner combining APB algorithm with L1,∞,∞ norm and layer filtering"""

    def __init__(self, pruning_type="unstructured"):
        self.pruning_type = pruning_type
        self.apb_optimizer = APBOptimizer()
        self.layer_filter = LayerDistanceCalculator()
        self.l1_infty_norm = L1InftyInftyNorm()

    def extract_layer_weights(self, model) -> List[torch.Tensor]:
        """Extract weights from prunable layers"""
        convs = model.get_prunable_layers(pruning_type=self.pruning_type)
        return [conv.conv.weight for conv in convs]

    def apb_prune(self, model, prune_rate: float):
        """Apply APB pruning algorithm"""
        convs = model.get_prunable_layers(pruning_type=self.pruning_type)
        device = next(model.parameters()).device

        # Collect all weights and compute global statistics
        all_weights = []
        all_importance = []

        for conv in convs:
            weight = conv.conv.weight
            importance = self.apb_optimizer.compute_importance_score(weight)
            all_weights.append(weight)
            all_importance.append(importance)

        # Compute global threshold for pruning
        all_importance_flat = torch.cat([imp.flatten() for imp in all_importance])
        global_threshold = torch.quantile(all_importance_flat, prune_rate / 100.0)

        # Apply APB to each layer
        for i, conv in enumerate(convs):
            weight = all_weights[i]
            importance = all_importance[i]

            # Create mask for weights below threshold
            prune_mask = importance < global_threshold

            # Select full-precision weights from remaining weights
            remaining_importance = importance * (~prune_mask).float()
            fp_mask = self.apb_optimizer.select_full_precision_weights(
                weight, remaining_importance
            )

            # Apply APB binarization
            processed_weight = self.apb_optimizer.binarize_weights(weight, fp_mask)

            # Apply pruning mask
            processed_weight = processed_weight * (~prune_mask).float()

            # Update layer weights
            conv.conv.weight.data = processed_weight

    def filter_layers_by_l1_infty_distance(self, model) -> List[int]:
        """Filter layers based on L1,∞,∞ norm distances"""
        layer_weights = self.extract_layer_weights(model)
        return self.layer_filter.filter_similar_layers(layer_weights)

    def apply_layer_filtering(self, model, selected_layer_indices: List[int]):
        """Apply layer filtering by zeroing out unselected layers"""
        convs = model.get_prunable_layers(pruning_type=self.pruning_type)

        for i, conv in enumerate(convs):
            if i not in selected_layer_indices:
                # Zero out this layer's weights
                conv.conv.weight.data.zero_()
                if hasattr(conv.conv, 'bias') and conv.conv.bias is not None:
                    conv.conv.bias.data.zero_()

    def compute_l1_infty_norm_statistics(self, model) -> Dict:
        """Compute L1,∞,∞ norm statistics for the model"""
        layer_weights = self.extract_layer_weights(model)
        layer_norms = []

        for weight in layer_weights:
            weight_np = weight.detach().cpu().numpy()
            norm_value = self.l1_infty_norm.norm(weight_np)
            layer_norms.append(norm_value)

        return {
            'layer_norms': layer_norms,
            'total_norm': sum(layer_norms),
            'mean_norm': np.mean(layer_norms),
            'std_norm': np.std(layer_norms),
            'max_norm': max(layer_norms),
            'min_norm': min(layer_norms)
        }

    def prune_and_optimize_pipeline(self, model, prune_rate: float,
                                    apply_layer_filtering: bool = True) -> Dict:
        """Complete pipeline: APB pruning -> layer filtering -> activation -> fine-tune ready"""

        print(f"Starting advanced pruning pipeline with {prune_rate}% pruning rate...")

        # Step 1: Apply APB pruning algorithm
        print("Step 1: Applying APB pruning...")
        self.apb_prune(model, prune_rate)

        # Compute statistics after pruning
        post_prune_stats = self.compute_l1_infty_norm_statistics(model)
        print(f"Post-pruning L1,∞,∞ norm statistics: {post_prune_stats}")

        # Step 2: Filter layers using L1,∞,∞ distance metric
        selected_layers = []
        if apply_layer_filtering:
            print("Step 2: Filtering similar layers using L1,∞,∞ distance...")
            selected_layers = self.filter_layers_by_l1_infty_distance(model)
            print(f"Selected {len(selected_layers)} layers out of {len(self.extract_layer_weights(model))}")

            # Apply layer filtering
            self.apply_layer_filtering(model, selected_layers)

        # Step 3: Prepare for activation and fine-tuning
        print("Step 3: Model ready for activation and fine-tuning...")

        # Compute final statistics
        final_stats = self.compute_l1_infty_norm_statistics(model)

        # Calculate sparsity
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        sparsity = zero_params / total_params

        results = {
            'selected_layers': selected_layers,
            'post_prune_stats': post_prune_stats,
            'final_stats': final_stats,
            'sparsity': sparsity,
            'total_params': total_params,
            'zero_params': zero_params
        }

        print(f"Pipeline completed. Final sparsity: {sparsity:.2%}")
        return results

    def prune_and_optimize(self, model, prune_rate: float, test_data=None):
        """Compatibility method with old interface"""
        return self.prune_and_optimize_pipeline(model, prune_rate, apply_layer_filtering=True)

    def prune(self, model, prune_rate: float):
        """Main pruning interface"""
        return self.prune_and_optimize_pipeline(model, prune_rate)
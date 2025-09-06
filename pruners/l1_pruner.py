import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import copy


class L1InftyInftyNorm:
    """L1,∞,∞ norm implementation - matches C++ reference exactly"""

    @staticmethod
    def norm(y: np.ndarray) -> float:
        if len(y.shape) == 1:
            return np.sum(np.abs(y))
        reshaped = y.reshape(y.shape[0], -1)
        return np.sum(np.max(np.abs(reshaped), axis=1))


class APBProcessor:
    """Automatic Prune Binarization (APB) processor with optional α, δ parameters"""

    def __init__(self, init_alpha=None, init_delta=None, binary_threshold: float = 0.1, fp_ratio: float = 0.05):
        """
        init_alpha/init_delta: nếu là số thì dùng trực tiếp, nếu None thì auto theo weight.
        binary_threshold: fallback threshold khi alpha/delta không khả dụng.
        fp_ratio: tỉ lệ giữ full-precision (chưa dùng nhiều ở bản alpha/delta, nhưng giữ để tương thích).
        """
        self.init_alpha = init_alpha if isinstance(init_alpha, (int, float)) else None
        self.init_delta = init_delta if isinstance(init_delta, (int, float)) else None

        self.binary_threshold = binary_threshold
        self.fp_ratio = fp_ratio

        self.layer_alphas: Dict[int, float] = {}
        self.layer_deltas: Dict[int, float] = {}

    def initialize_layer_parameters(self, weight: torch.Tensor, layer_id: int):
        if layer_id not in self.layer_alphas:
            weight_abs = torch.abs(weight)
            # safe item() to float
            alpha = self.init_alpha if self.init_alpha is not None else float(torch.mean(weight_abs).item())
            delta = self.init_delta if self.init_delta is not None else float(3.0 * torch.std(weight_abs).item())
            self.layer_alphas[layer_id] = alpha
            self.layer_deltas[layer_id] = delta

    def apply_apb_to_layer(self, weight: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Áp dụng APB cho 1 layer"""
        self.initialize_layer_parameters(weight, layer_id)

        alpha = self.layer_alphas.get(layer_id, None)
        delta = self.layer_deltas.get(layer_id, None)

        # nếu alpha/delta tồn tại, dùng alpha+delta, nếu không, fallback binary_threshold
        binarization_threshold = (alpha + delta) if (alpha is not None and delta is not None) else self.binary_threshold

        weight_magnitude = torch.abs(weight)
        binarize_mask = weight_magnitude <= binarization_threshold

        # Nếu alpha is None (edge), tránh nhân với None
        alpha_val = alpha if alpha is not None else self.binary_threshold

        binary_part = torch.sign(weight) * float(alpha_val) * binarize_mask.float()
        fp_part = weight * (~binarize_mask).float()

        return binary_part + fp_part

    def compute_apb_statistics(self, weight: torch.Tensor, layer_id: int) -> Dict:
        alpha = self.layer_alphas.get(layer_id, None)
        delta = self.layer_deltas.get(layer_id, None)
        threshold = (alpha + delta) if (alpha is not None and delta is not None) else self.binary_threshold

        weight_magnitude = torch.abs(weight)
        binarize_mask = weight_magnitude <= threshold

        binary_count = int(binarize_mask.sum().item())
        fp_count = int((~binarize_mask).sum().item())
        total_count = weight.numel()

        return {
            "alpha": alpha,
            "delta": delta,
            "threshold": threshold,
            "binary_ratio": binary_count / total_count if total_count > 0 else 0.0,
            "fp_ratio": fp_count / total_count if total_count > 0 else 0.0,
            "binary_count": binary_count,
            "fp_count": fp_count,
        }

    def update_layer_parameters(self, layer_id: int, new_alpha: float = None, new_delta: float = None):
        if new_alpha is not None:
            self.layer_alphas[layer_id] = new_alpha
        if new_delta is not None:
            self.layer_deltas[layer_id] = new_delta

    # Legacy support
    def compute_importance_score(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.abs(weight)

    def binarize_weights(self, weight: torch.Tensor, fp_mask: torch.Tensor) -> torch.Tensor:
        binary_part = torch.sign(weight) * (1 - fp_mask.float())
        fp_part = weight * fp_mask.float()
        return binary_part + fp_part


class LayerSimilarityFilter:
    """Lọc layers tương tự dựa trên L1,∞,∞ norm"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.l1_infty_norm = L1InftyInftyNorm()

    def extract_layer_features(self, weight: torch.Tensor) -> np.ndarray:
        weight_np = weight.detach().cpu().numpy()
        flat_weight = weight_np.flatten()
        features = [
            np.mean(np.abs(flat_weight)),
            np.std(flat_weight),
            np.percentile(np.abs(flat_weight), 90),
            np.percentile(np.abs(flat_weight), 95),
            self.l1_infty_norm.norm(weight_np),
            np.max(np.abs(flat_weight)),
            np.count_nonzero(flat_weight) / len(flat_weight) if len(flat_weight) > 0 else 0.0
        ]
        return np.array(features)

    def compute_layer_similarities(self, layer_weights: List[torch.Tensor]) -> np.ndarray:
        n_layers = len(layer_weights)
        if n_layers <= 1:
            return np.array([[1.0]])
        features_list = [self.extract_layer_features(w) for w in layer_weights]
        feature_matrix = np.stack(features_list)
        similarity_matrix = cosine_similarity(feature_matrix)
        return similarity_matrix

    def select_diverse_layers(self, layer_weights: List[torch.Tensor]) -> List[int]:
        n_layers = len(layer_weights)
        if n_layers <= 1:
            return list(range(n_layers))

        similarity_matrix = self.compute_layer_similarities(layer_weights)
        selected_layers = []
        remaining_layers = list(range(n_layers))

        layer_norms = [self.l1_infty_norm.norm(w.detach().cpu().numpy()) for w in layer_weights]
        first_layer = int(np.argmax(layer_norms))
        selected_layers.append(first_layer)
        remaining_layers.remove(first_layer)

        while remaining_layers:
            best_layer = None
            min_max_similarity = float('inf')

            for candidate in remaining_layers:
                max_similarity = 0.0
                for selected in selected_layers:
                    max_similarity = max(max_similarity, float(similarity_matrix[candidate, selected]))
                if max_similarity < min_max_similarity:
                    min_max_similarity = max_similarity
                    best_layer = candidate

            if min_max_similarity > self.similarity_threshold:
                break

            if best_layer is not None:
                selected_layers.append(best_layer)
                remaining_layers.remove(best_layer)

        return sorted(selected_layers)


class TensorToFineTuneReady:
    """
    Pipeline: Tensor → APB → L1,∞,∞ Layer Filtering → Fine-tune Ready Model
    """

    def __init__(self, pruning_type: str = "unstructured", init_alpha: Optional[float] = None,
                 init_delta: Optional[float] = None, similarity_threshold: float = 0.85):
        # Lưu pruning_type (để gọi get_prunable_layers(pruning_type=...))
        self.pruning_type = pruning_type
        self.apb_processor = APBProcessor(init_alpha=init_alpha, init_delta=init_delta)
        self.layer_filter = LayerSimilarityFilter(similarity_threshold=similarity_threshold)
        self.l1_infty_norm = L1InftyInftyNorm()

    def process_model(self, model, prune_rate: float = 50.0) -> Dict:
        print(f"🚀 Starting pipeline (pruning_type={self.pruning_type}) prune_rate={prune_rate}%")
        apb_stats = self._apply_apb_to_all_layers(model)
        filtering_stats = self._filter_similar_layers(model)
        final_stats = self._compute_final_statistics(model)

        results = {
            'apb_stats': apb_stats,
            'filtering_stats': filtering_stats,
            'final_stats': final_stats,
            'pipeline_success': True
        }
        return results

    def _apply_apb_to_all_layers(self, model) -> Dict:
        convs = model.get_prunable_layers(pruning_type=self.pruning_type)
        n = len(convs)
        print(f"   Found {n} prunable layers")

        processed_layers = 0
        total_original_params = 0
        total_remaining_params = 0

        for i, conv in enumerate(convs):
            weight = conv.conv.weight
            total_original_params += weight.numel()

            processed_weight = self.apb_processor.apply_apb_to_layer(weight, i)
            conv.conv.weight.data = processed_weight

            remaining_params = int(torch.count_nonzero(processed_weight).item())
            total_remaining_params += remaining_params

            processed_layers += 1
            if (i + 1) % 10 == 0 or (i + 1) == n:
                print(f"   Processed {i + 1}/{n} layers...")

        apb_sparsity = 0.0
        if total_original_params > 0:
            apb_sparsity = 1.0 - (total_remaining_params / total_original_params)

        stats = {
            'processed_layers': processed_layers,
            'original_params': total_original_params,
            'remaining_params': total_remaining_params,
            'apb_sparsity': apb_sparsity
        }
        print(f"   ✅ APB completed: {apb_sparsity:.2%} sparsity achieved")
        return stats

    def _filter_similar_layers(self, model) -> Dict:
        convs = model.get_prunable_layers(pruning_type=self.pruning_type)
        layer_weights = [conv.conv.weight for conv in convs]
        original_layer_count = len(layer_weights)
        print(f"   Analyzing {original_layer_count} layers for similarity...")

        selected_layer_indices = self.layer_filter.select_diverse_layers(layer_weights)
        layers_to_remove = [i for i in range(original_layer_count) if i not in selected_layer_indices]

        print(f"   Selected {len(selected_layer_indices)} diverse layers")
        print(f"   Removing {len(layers_to_remove)} similar layers")

        for remove_idx in layers_to_remove:
            conv = convs[remove_idx]
            conv.conv.weight.data.zero_()
            if hasattr(conv.conv, 'bias') and conv.conv.bias is not None:
                conv.conv.bias.data.zero_()

        remaining_norms = []
        for idx in selected_layer_indices:
            weight = layer_weights[idx]
            weight_np = weight.detach().cpu().numpy()
            norm_val = self.l1_infty_norm.norm(weight_np)
            remaining_norms.append(norm_val)

        layer_reduction_ratio = 0.0
        if original_layer_count > 0:
            layer_reduction_ratio = len(layers_to_remove) / original_layer_count

        stats = {
            'original_layers': original_layer_count,
            'remaining_layers': len(selected_layer_indices),
            'removed_layers': len(layers_to_remove),
            'selected_indices': selected_layer_indices,
            'removed_indices': layers_to_remove,
            'layer_reduction_ratio': layer_reduction_ratio,
            'remaining_layer_norms': remaining_norms,
            'avg_remaining_norm': float(np.mean(remaining_norms)) if remaining_norms else 0.0
        }
        print(f"   ✅ Layer filtering completed: {stats['layer_reduction_ratio']:.2%} layers removed")
        return stats

    def _compute_final_statistics(self, model) -> Dict:
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum(int((p == 0).sum().item()) for p in model.parameters())
        active_params = total_params - zero_params
        sparsity = (zero_params / total_params) if total_params > 0 else 0.0

        convs = model.get_prunable_layers(pruning_type=self.pruning_type)
        active_layers = 0
        layer_norms = []
        for conv in convs:
            weight = conv.conv.weight
            if int(torch.count_nonzero(weight).item()) > 0:
                active_layers += 1
                weight_np = weight.detach().cpu().numpy()
                layer_norms.append(self.l1_infty_norm.norm(weight_np))

        stats = {
            'total_params': total_params,
            'active_params': active_params,
            'zero_params': zero_params,
            'sparsity': sparsity,
            'active_layers': active_layers,
            'total_layers': len(convs),
            'layer_norms': layer_norms,
            'total_l1_infty_norm': float(sum(layer_norms)),
            'avg_layer_norm': float(np.mean(layer_norms)) if layer_norms else 0.0,
            'model_ready_for_finetuning': True
        }
        return stats

    # compatibility: some callers expect prune_and_optimize
    def prune_and_optimize(self, model, prune_rate: float = 50.0):
        return self.process_model(model, prune_rate)

    def quick_process(self, model, prune_rate: float = 50.0):
        results = self.process_model(model, prune_rate)
        return model, results

import torch
import torch.nn as nn
import numpy as np
import heapq
from typing import List

class PITOOptimizer:
    """Projection Inverse Total Order algorithm implementation"""

    def __init__(self, m: int, n: int, C: float = 1.0):
        self.m = m  # number of constraints
        self.n = n  # number of variables
        self.C = C  # regularization parameter
        self.theta = 0

    def update_theta(self):
        self.theta += 0.01  # Simple update rule

    def projection_inverse_total_order(self, Y: np.ndarray) -> np.ndarray:
        # Lấy ma trận dấu để khôi phục sau
        sign_matrix = np.sign(Y)
        Y_abs = np.abs(Y)

        S = np.sum(Y_abs, axis=0)
        P = []
        for j, s in enumerate(S):
            heapq.heappush(P, (-s, j))

        k = np.ones(self.m, dtype=int) * (self.n + 1)
        a = np.zeros(self.m)

        theta = 0
        theta_changed = True
        iterations = 0

        while theta_changed and iterations < 100:
            theta_changed = False
            iterations += 1

            while P:
                neg_val, j = heapq.heappop(P)
                top_val = -neg_val
                i = k[j]

                k[j] = k[j] - 1

                if i == self.n + 1:
                    a[j] = 1
                    self.update_theta()

                    if np.linalg.norm(Y_abs[:, j]) < self.theta:
                        a[j] = 0
                        self.update_theta()
                        break

                    Y_abs[:, j] = np.sort(Y_abs[:, j])

                else:
                    if 0 <= k[j] < Y_abs.shape[0]:
                        S[j] = S[j] - Y_abs[k[j], j]

                    self.update_theta()

                    if (
                        k[j] >= 0
                        and k[j] > 0
                        and (S[j] - self.theta) / k[j] < Y_abs[min(i, Y_abs.shape[0]-1), j]
                    ):
                        k[j] = k[j] + 1
                        if k[j] < Y_abs.shape[0]:
                            S[j] = S[j] + Y_abs[k[j], j]
                        self.update_theta()
                        break

                if S[j] != 0:
                    heapq.heappush(P, (-S[j], j))

        X = np.zeros_like(Y_abs)
        for i in range(Y_abs.shape[0]):
            for j in range(Y_abs.shape[1]):
                if k[j] > 0:
                    X[i, j] = min(Y_abs[i, j], max(0, (S[j] - self.theta) / k[j]))
                else:
                    X[i, j] = max(0, Y_abs[i, j])

        # Khôi phục lại dấu
        return X * sign_matrix

class L1Pruner:
    def __init__(self, pruning_type="unstructured"):
        self.pruning_type = pruning_type
        self.pito_optimizer = None

    def structured_prune(self, model, prune_rate):
        convs = model.get_prunable_layers(pruning_type=self.pruning_type)

        channel_norms = []
        for conv in convs:
            channel_norms.append(
                torch.sum(
                    torch.abs(conv.conv.weight.view(conv.conv.out_channels, -1)), axis=1
                )
            )

        threshold = np.percentile(channel_norms, prune_rate)

        for conv in convs:
            channel_norms = torch.sum(
                torch.abs(conv.conv.weight.view(conv.conv.out_channels, -1)), axis=1
            )
            mask = (channel_norms >= threshold).float()
            conv.mask = torch.einsum("cijk,c->cijk", conv.conv.weight.data, mask)

    def unstructured_prune(self, model, prune_rate=50.0):
        convs = model.get_prunable_layers(pruning_type=self.pruning_type)
        device = next(model.parameters()).device
        all_weights = torch.tensor([], device=device)

        for conv in convs:
            all_weights = torch.cat((all_weights.view(-1), conv.conv.weight.view(-1)))

        abs_weights = torch.abs(all_weights.detach())
        threshold = np.percentile(abs_weights.detach().cpu().numpy(), prune_rate)

        for conv in convs:
            mask_update = torch.gt(torch.abs(conv.conv.weight), threshold).float().to(device)
            mask_weight = conv.mask.mask.weight.to(device)
            conv.mask.update(mask_update * mask_weight)

    def extract_sparse_matrix(self, model):
        sparse_matrices = []
        convs = model.get_prunable_layers(pruning_type=self.pruning_type)

        for conv in convs:
            weight = conv.conv.weight.data
            weight_2d = weight.view(weight.shape[0], -1)
            sparse_matrices.append(weight_2d.cpu().numpy())

        if sparse_matrices:
            return np.concatenate(sparse_matrices, axis=0)
        return np.array([])

    def apply_pito_to_sparse_matrix(self, sparse_matrix):
        """Apply PITO nhưng giữ lại dấu"""
        if sparse_matrix.size == 0:
            return sparse_matrix

        m, n = sparse_matrix.shape
        if self.pito_optimizer is None:
            self.pito_optimizer = PITOOptimizer(m=m, n=n)

        optimized_matrix = self.pito_optimizer.projection_inverse_total_order(sparse_matrix)
        return optimized_matrix

    def sparse_then_pito(self, model):
        sparse_matrix = self.extract_sparse_matrix(model)
        print(f"Extracted sparse matrix shape: {sparse_matrix.shape}")
        print(f"Sparsity: {(sparse_matrix == 0).sum() / sparse_matrix.size:.2%}")

        optimized_matrix = self.apply_pito_to_sparse_matrix(sparse_matrix)
        print(f"PITO optimized matrix shape: {optimized_matrix.shape}")

        return sparse_matrix, optimized_matrix

    def prune_and_optimize(self, model, prune_rate, test_data=None):
        self.prune(model, prune_rate)

        if test_data is not None:
            model.eval()
            with torch.no_grad():
                predictions = model(test_data)
                optimized_predictions = self.apply_pito_to_sparse_matrix(predictions.cpu().numpy())
                return optimized_predictions
        return None

    def prune(self, model, prune_rate):
        if self.pruning_type.lower() == "unstructured":
            self.unstructured_prune(model, prune_rate)
        elif self.pruning_type.lower() == "structured":
            self.structured_prune(model, prune_rate)
        else:
            raise ValueError("Invalid type of pruning")

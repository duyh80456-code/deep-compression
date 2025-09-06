from .l1_pruner import TensorToFineTuneReady

# Alias để giữ tên cũ L1Pruner
L1Pruner = TensorToFineTuneReady

__all__ = ["L1Pruner", "get_pruner"]


def get_pruner(pruner, pruning_type, **kwargs):

    if pruner == "L1Pruner":
        return L1Pruner(pruning_type=pruning_type, **kwargs)
    else:
        raise NotImplementedError(
            f"Pruner '{pruner}' is not implemented. Available: ['L1Pruner']"
        )

from .l1_pruner import TensorToFineTuneReady

# Alias để giữ tên cũ L1Pruner
L1Pruner = TensorToFineTuneReady

__all__ = ["L1Pruner", "get_pruner"]


def get_pruner(pruner, pruning_type, **kwargs):
    """
    Factory trả về đúng pruner theo tên.

    Args:
        pruner (str): Tên pruner, ví dụ "L1Pruner".
        pruning_type (str): Loại pruning, ví dụ "unstructured".
        **kwargs: Các tham số khác (vd. init_alpha, init_delta) sẽ truyền vào L1Pruner nếu cần.
    """
    if pruner == "L1Pruner":
        return L1Pruner(pruning_type=pruning_type, **kwargs)
    else:
        raise NotImplementedError(
            f"Pruner '{pruner}' is not implemented. Available: ['L1Pruner']"
        )

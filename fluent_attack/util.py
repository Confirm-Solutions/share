import contextlib
from typing import Callable, List, Tuple

import torch
import torch.distributions


def mellowmax(t: torch.Tensor, alpha=1.0, dim=-1):
    return (
        1.0
        / alpha
        * (
            torch.logsumexp(alpha * t, dim=dim)
            - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device))
        )
    )


@contextlib.contextmanager
def add_fwd_hooks(module_hooks: List[Tuple[torch.nn.Module, Callable]]):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for mod, hk in module_hooks:
            handles.append(mod.register_forward_hook(hk))
        yield
    finally:
        for h in handles:
            h.remove()

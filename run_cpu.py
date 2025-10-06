# run_cpu.py — force CPU even if the code calls .cuda() or .to("cuda")
import os, sys, runpy
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # pretend there are no GPUs
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # harmless; allows CPU fallback

import torch  # noqa: E402

# --- helpers ---------------------------------------------------------------
def _is_cuda_arg(arg):
    import torch as _t
    if isinstance(arg, str):
        return arg.lower().startswith("cuda")
    if isinstance(arg, _t.device):
        return arg.type.startswith("cuda")
    return False

def _filter_to_args(args, kwargs):
    # replace any CUDA device with CPU
    new_args = list(args)
    if new_args and _is_cuda_arg(new_args[0]):
        new_args[0] = torch.device("cpu")
    if "device" in kwargs and _is_cuda_arg(kwargs["device"]):
        kwargs = dict(kwargs)
        kwargs["device"] = torch.device("cpu")
    return tuple(new_args), kwargs
# ---------------------------------------------------------------------------

# 1) Make torch report no CUDA
def _false(): return False
torch.cuda.is_available = _false  # type: ignore[attr-defined]

# 2) Neutralize .cuda() on tensors and modules
def _noop(self, *a, **k): return self
torch.Tensor.cuda = _noop           # type: ignore[attr-defined]
torch.nn.Module.cuda = _noop        # type: ignore[attr-defined]

# 3) Make Module.to(...) ignore CUDA targets
_old_mod_to = torch.nn.Module.to
def _safe_mod_to(self, *args, **kwargs):
    args, kwargs = _filter_to_args(args, kwargs)
    return _old_mod_to(self, *args, **kwargs)
torch.nn.Module.to = _safe_mod_to   # type: ignore[attr-defined]

# 4) Make Tensor.to(...) ignore CUDA targets
_old_ten_to = torch.Tensor.to
def _safe_ten_to(self, *args, **kwargs):
    args, kwargs = _filter_to_args(args, kwargs)
    return _old_ten_to(self, *args, **kwargs)
torch.Tensor.to = _safe_ten_to      # type: ignore[attr-defined]

# 5) Forward all CLI args to the original entry script
sys.argv = ["train_net.py"] + sys.argv[1:]
runpy.run_path("train_net.py", run_name="__main__")

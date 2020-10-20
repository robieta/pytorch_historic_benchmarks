import torch

def make_globals(stmt: str):
    if "# TS: script fn `x + 1`" in stmt:
        def fn(y: torch.Tensor):
            return y + 1

        model = torch.jit.script(fn)

        # JIT doesn't like imports before the model definition.
        from torch.utils.benchmark import CopyIfCallgrind
        return {"model": CopyIfCallgrind(model)}

    elif "# TS: script Module `x + 1`" in stmt:
        class Fn(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = torch.jit.script(Fn())

        # JIT doesn't like imports before the model definition.
        from torch.utils.benchmark import CopyIfCallgrind
        return {"model": CopyIfCallgrind(model)}

    elif "# TS:" in stmt:
        raise NotImplementedError(f"Unknown stmt which expects TorchScript: {stmt}")
    return {}

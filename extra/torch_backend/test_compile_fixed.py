# ruff: noqa: E731
import torch, time
from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_module_simplified
from torch._decomp import get_decompositions
from extra.torch_backend.backend import decomps

@register_backend
def tiny(gm:torch.fx.GraphModule, sample_inputs):
  def fw_compiler(gm:torch.fx.GraphModule, sample_inputs):
    def run(*args:torch.Tensor):
      # move inputs to tiny device, run graph, move outputs back
      tiny_args = tuple(a.to("tiny") if a.is_cpu else a for a in args)
      outs = gm(*tiny_args)
      if not isinstance(outs, (list, tuple)): outs = (outs,)
      return tuple(o.cpu() if isinstance(o, torch.Tensor) and str(o.device).startswith("tiny") else o for o in outs)
    return run
  return aot_module_simplified(gm, sample_inputs, decompositions=get_decompositions(decomps), fw_compiler=fw_compiler)

if __name__ == "__main__":
  import extra.torch_backend.backend  # noqa: F401
  import numpy as np

  def check(name, fn):
    try:
      fn()
      print(f"  PASS  {name}")
    except Exception as e:
      print(f"  FAIL  {name}: {e}")

  def bench(name, fn, warmup=3, iters=10):
    for _ in range(warmup): fn()
    st = time.perf_counter()
    for _ in range(iters): fn()
    print(f"  {name}: {(time.perf_counter()-st)/iters*1000:.1f}ms/iter")

  print("--- torch.compile(backend='tiny') tests ---")

  # dynamo traces on CPU tensors, compiled fn moves to tiny device via aot_autograd
  def test_sincos():
    def f(x, y): return torch.sin(x) + torch.cos(y)
    cf = torch.compile(f, backend="tiny")
    x, y = torch.randn(100, 100), torch.randn(100, 100)
    ref = f(x, y).numpy()
    for _ in range(3): cf(x, y)
    np.testing.assert_allclose(ref, cf(x, y).numpy(), rtol=1e-4, atol=1e-4)
  check("sin+cos fusion", test_sincos)

  def test_matmul():
    def f(x, w1, w2): return torch.relu(torch.relu(x @ w1) @ w2)
    cf = torch.compile(f, backend="tiny")
    x, w1, w2 = torch.randn(64, 128), torch.randn(128, 256), torch.randn(256, 64)
    ref = f(x, w1, w2).numpy()
    for _ in range(3): cf(x, w1, w2)
    np.testing.assert_allclose(ref, cf(x, w1, w2).numpy(), rtol=1e-3, atol=1e-3)
  check("matmul chain", test_matmul)

  def test_mlp():
    def f(x, w1, b1, w2, b2): return torch.relu(torch.relu(x @ w1 + b1) @ w2 + b2)
    cf = torch.compile(f, backend="tiny")
    x = torch.randn(32, 256)
    w1, b1 = torch.randn(256, 512), torch.randn(512)
    w2, b2 = torch.randn(512, 256), torch.randn(256)
    ref = f(x, w1, b1, w2, b2).numpy()
    for _ in range(3): cf(x, w1, b1, w2, b2)
    np.testing.assert_allclose(ref, cf(x, w1, b1, w2, b2).numpy(), rtol=1e-3, atol=1e-3)
  check("MLP block", test_mlp)

  print("\n--- benchmark ---")
  x = torch.randn(32, 256)
  w1, b1 = torch.randn(256, 512), torch.randn(512)
  w2, b2 = torch.randn(512, 256), torch.randn(256)
  def mlp(x): return torch.relu(torch.relu(x @ w1 + b1) @ w2 + b2)
  compiled_mlp = torch.compile(mlp, backend="tiny")
  bench("eager (cpu)", lambda: mlp(x))
  bench("compiled", lambda: compiled_mlp(x), warmup=5)

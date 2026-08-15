"""
Shared pytest fixtures and configuration for KVGenius tests.
"""
import sys
import os
import types

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Exclude the legacy integration-test script that requires actual GPU hardware
collect_ignore = ["test_chat.py"]


def _make_torch_mock():
    """Create a minimal mock of the torch module."""
    torch_mock = types.ModuleType("torch")

    # Tensor-like class for tests
    class FakeTensor:
        def __init__(self, data=None):
            self.data = data or []
            self.shape = (1, len(self.data) if self.data else 0)

        def to(self, device):
            return self

        def __getitem__(self, idx):
            return self

        def __len__(self):
            return len(self.data)

    torch_mock.Tensor = FakeTensor
    torch_mock.no_grad = lambda: _NoGradContext()
    torch_mock.float16 = "float16"
    torch_mock.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        memory_allocated=lambda n=0: 0,
    )
    torch_mock.device = lambda x: x

    class _NoGradContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    return torch_mock


def _make_pil_mock():
    """Create a minimal mock of the PIL (Pillow) package."""
    pil_mock = types.ModuleType("PIL")

    class FakeImage:
        def __init__(self, info=None):
            self.info = info or {}

        @staticmethod
        def open(filepath):
            return FakeImage()

    image_mod = types.ModuleType("PIL.Image")
    image_mod.Image = FakeImage
    image_mod.open = FakeImage.open

    png_mod = types.ModuleType("PIL.PngImagePlugin")
    png_mod.PngInfo = object

    pil_mock.Image = image_mod
    pil_mock.PngImagePlugin = png_mod

    return pil_mock, image_mod, png_mod


# Install torch and PIL mocks so modules that import them at the top level
# can be imported without having those heavy packages installed.
if "torch" not in sys.modules:
    sys.modules["torch"] = _make_torch_mock()

if "PIL" not in sys.modules:
    pil_mock, image_mod, png_mod = _make_pil_mock()
    sys.modules["PIL"] = pil_mock
    sys.modules["PIL.Image"] = image_mod
    sys.modules["PIL.PngImagePlugin"] = png_mod

# Stub out heavy ML libraries that may be pulled in transitively.
# Provide enough of an API surface that imports don't fail.
for _heavy in ("accelerate", "diffusers", "safetensors", "bitsandbytes", "peft"):
    if _heavy not in sys.modules:
        sys.modules[_heavy] = types.ModuleType(_heavy)

# transformers needs a richer stub so that test_chat.py (legacy integration
# test) can at least be collected without errors.
if "transformers" not in sys.modules:
    _transformers = types.ModuleType("transformers")

    class _StubClass:
        """Generic stub that accepts any constructor arguments."""
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _transformers.AutoModelForCausalLM = _StubClass
    _transformers.AutoTokenizer = _StubClass
    _transformers.AutoModelForSeq2SeqLM = _StubClass
    _transformers.pipeline = lambda *a, **kw: None
    sys.modules["transformers"] = _transformers

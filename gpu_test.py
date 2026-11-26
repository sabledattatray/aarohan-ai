
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA GPU not available")

print("GPU:", torch.cuda.get_device_name(0))

props = torch.cuda.get_device_properties(0)
print("Total VRAM:", round(props.total_memory / 1024**3, 2), "GB")

# GPU computation test
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")

c = a @ b
torch.cuda.synchronize()

print("Matrix multiplication:", c.shape)
print("GPU test: PASSED")
print("Allocated VRAM:", round(
    torch.cuda.memory_allocated() / 1024**3, 3
), "GB")

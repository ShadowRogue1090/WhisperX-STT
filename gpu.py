import torch

print("Torch version:")
print(torch.__version__)

print("HIP:")
print(torch.version.hip)

print("CUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

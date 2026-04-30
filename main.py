import torch_opencl

print("=== Device Info ===")
print(f"is_available     : {torch_opencl.is_available()}")
print(f"device_count     : {torch_opencl.device_count()}")
print(f"current_device   : {torch_opencl.current_device()}")
print(f"device_name      : {torch_opencl.device_name(0)}")
print(f"device_properties: {torch_opencl.get_device_properties(0)}")

print()
print("=== Device Switch ===")
torch_opencl.set_device(0)
print(f"after set_device(0): {torch_opencl.current_device()}")

print()
print("=== Synchronize ===")
torch_opencl.synchronize(0)
print("synchronize OK")

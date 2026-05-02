import torch
import torch_opencl

cpu = torch.tensor([1.0, 2.0, 3.0])
print("CPU:", cpu)
print("CPU data_ptr:", cpu.data_ptr())

# Host → Device
cl = cpu.to("opencl")
print("OpenCL device:", cl.device)
print("OpenCL data_ptr:", cl.data_ptr())

# Pastikan data_ptr berbeda (beda storage)
assert cpu.data_ptr() != cl.data_ptr(), "data_ptr sama — tidak di-copy!"

# Modifikasi tensor CPU setelah copy
cpu[0] = 999.0
print("CPU setelah modifikasi:", cpu)

# Device → Host
back = cl.cpu()
print("Back:", back)

# Kalau copy benar, back[0] harus tetap 1.0, bukan 999.0
assert back[0].item() == 1.0, f"Nilai salah: {back[0].item()} (harusnya 1.0)"
assert torch.allclose(torch.tensor([1.0, 2.0, 3.0]), back), "Nilai tidak cocok!"
print("OK — data benar-benar di-copy ke device dan kembali dengan benar")

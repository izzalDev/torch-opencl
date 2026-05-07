# tests/conftest.py
import pyopencl as cl


def pytest_configure(config):
    """Print OpenCL environment info at the start of test session."""
    print("\n=== OpenCL Environment ===")
    try:
        platforms = cl.get_platforms()
        if not platforms:
            print("No OpenCL platforms found")
            return

        for i, platform in enumerate(platforms):
            print(f"Platform {i}: {platform.name}")
            print(f"  Vendor  : {platform.vendor}")
            print(f"  Version : {platform.version}")

            for device_type, label in [
                (cl.device_type.GPU, "GPU"),
                (cl.device_type.CPU, "CPU"),
            ]:
                try:
                    devices = platform.get_devices(device_type=device_type)
                    for j, device in enumerate(devices):
                        print(f"  {label} {j}: {device.name}")
                except cl.LogicError:
                    print(f"  {label} : none")

    except cl.Error as e:
        print(f"OpenCL not available: {e}")
    finally:
        print("==========================\n")

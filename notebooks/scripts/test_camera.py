"""Test the RealSense D435i camera connection."""
try:
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) > 0:
        dev = devices[0]
        print(f"Camera found: {dev.get_info(rs.camera_info.name)}")
        print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
    else:
        print("No RealSense camera detected "
              "(connect D435i or use recorded data)")
except ImportError:
    print("pyrealsense2 not installed")
except Exception as e:
    print(f"Camera check failed: {e}")

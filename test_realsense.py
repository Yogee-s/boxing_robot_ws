import pyrealsense2 as rs
import time

def main():
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No devices found")
        return

    print("Found devices:")
    for d in ctx.devices:
        print(f"  {d.get_info(rs.camera_info.name)} - {d.get_info(rs.camera_info.serial_number)}")

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        
        print("Starting pipeline...")
        pipeline.start(config)
        print("Pipeline started successfully!")
        
        for i in range(30):
            frames = pipeline.wait_for_frames()
            print(".", end="", flush=True)
            
        pipeline.stop()
        print("\nPipeline stopped.")
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()

import pyrealsense2 as rs

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()
pipeline.start(config)

# Get the device that is connected to the pipeline
profile = pipeline.get_active_profile()
device = profile.get_device()

# Get the depth sensor of the connected device
depth_sensor = device.first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Get the stream profile and extract camera intrinsics
stream_profile = profile.get_stream(rs.stream.depth)
intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()

print("Camera Intrinsics:", intrinsics)

# Stop the pipeline
pipeline.stop()
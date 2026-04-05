# ROS2 Topic Draft

## Inputs
- `/camera/sonar/image` (sensor_msgs/Image)

## Outputs
- `/def_nasdetr/detections` (custom detection array)

## Notes
- Use stable message schema with normalized bbox + confidence + class id.
- Preserve frame timestamp for downstream fusion.

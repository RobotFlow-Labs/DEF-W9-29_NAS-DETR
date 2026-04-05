# Production Checklist — DEF-nasdetr

- [ ] Validate URPC2021/URPC2022 data mounts on server.
- [ ] Re-run full training with paper A1/A2 settings.
- [ ] Export ONNX and TensorRT FP16/FP32 variants.
- [ ] Measure end-to-end FPS and compare with paper §4.5.
- [ ] Add runtime monitoring, retries, and model health probes.
- [ ] Validate ROS2 integration under live message load.

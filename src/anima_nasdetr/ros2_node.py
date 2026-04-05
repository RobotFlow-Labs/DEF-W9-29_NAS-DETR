"""ROS2 node scaffold for DEF-nasdetr.

This file intentionally avoids importing rclpy so local environments without ROS2
can still run tests and packaging. Replace stubs with real ROS2 node wiring on
deployment host.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Ros2TopicConfig:
    image_topic: str = "/camera/sonar/image"
    detection_topic: str = "/def_nasdetr/detections"


class NASDETRNodeScaffold:
    def __init__(self, topic_cfg: Ros2TopicConfig | None = None) -> None:
        self.topic_cfg = topic_cfg or Ros2TopicConfig()

    def process_image(self, _msg) -> dict:
        return {
            "topic_out": self.topic_cfg.detection_topic,
            "detections": [],
            "note": "replace with model invocation on ROS2 runtime",
        }

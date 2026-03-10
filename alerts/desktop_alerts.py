import subprocess
import sys
from loguru import logger
from plyer import notification
import yaml
import os

class DesktopAlert:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.enabled = self.config['alerts'].get('desktop_enabled', False)

    def send(self, title, message):
        if not self.enabled:
            return

        try:
            notification.notify(
                title=title,
                message=message,
                app_name="ForexVision",
                timeout=10
            )
            logger.info(f"Desktop alert sent: {title}")
        except Exception as e:
            logger.error(f"Failed to send desktop alert: {e}")

    def send_critical(self, message):
        self.send("ForexVision Critical", message)

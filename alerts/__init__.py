"""
Alerts Module

Contains notification systems for email and desktop alerts based on
trading signals and system status.
"""

from .email_alerts import EmailAlert
from .desktop_alerts import DesktopAlert

__all__ = [
    "EmailAlert",
    "DesktopAlert"
]

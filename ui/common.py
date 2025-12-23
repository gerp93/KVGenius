"""
Common Flet imports and helper functions for KVGenius UI.
"""
import logging
import sys

import flet as ft
from flet import (
    Page, Text, Column, Row, Container, Card, Tabs, Tab,
    TextField, ElevatedButton, ProgressBar, Image, Dropdown,
    Slider, IconButton, ListView, Colors, Icons,
    MainAxisAlignment, CrossAxisAlignment, ScrollMode,
    dropdown, SnackBar, AlertDialog, TextButton, FontWeight,
    padding, border_radius, border, GridView, Checkbox, Stack, alignment,
    ControlState, ButtonStyle, Divider,
)

# Set up logging
logger = logging.getLogger(__name__)


def debug_print(msg):
    """Print with immediate flush."""
    sys.stdout.write(f"[DEBUG] {msg}\n")
    sys.stdout.flush()


def get_gpu_info():
    """Get GPU name and compute capability for display."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            return f"{gpu_name} (sm_{major}{minor})"
        else:
            return "CPU (No GPU detected)"
    except:
        return "GPU detection unavailable"

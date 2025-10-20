"""
Utility functions for configuration and logging.
"""
import os
import yaml
import json
import logging
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "model": {
            "name": "microsoft/DialoGPT-medium",
            "cache_dir": "./model_cache",
            "device": "auto"
        },
        "generation": {
            "max_length": 1000,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        },
        "chat": {
            "max_history": 5,
            "system_prompt": "You are a helpful AI assistant."
        },
        "app": {
            "debug": False,
            "log_level": "INFO"
        }
    }


def setup_logging(level: str = "INFO"):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_environment():
    """Load environment variables from .env file."""
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        logging.info("Environment variables loaded from .env")
    else:
        logging.info("No .env file found, using system environment")


def save_chat_history(history: list, filepath: str = "chat_history.json"):
    """
    Save chat history to JSON file.
    
    Args:
        history: List of conversation messages
        filepath: Path to save history file
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        logging.info(f"Chat history saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving chat history: {e}")


def load_chat_history(filepath: str = "chat_history.json") -> list:
    """
    Load chat history from JSON file.
    
    Args:
        filepath: Path to history file
        
    Returns:
        List of conversation messages
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Error loading chat history: {e}")
    return []

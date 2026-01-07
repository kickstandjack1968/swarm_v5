"""
Configuration module for color settings.
"""

from typing import Optional, Dict, Any


class Settings:
    """
    Configuration settings for text colors.
    
    Attributes:
        text_color (str): Color code for the text
        year_color (str): Color code for the year
    """
    
    def __init__(self, text_color: str = "white", year_color: str = "yellow"):
        self.text_color = text_color
        self.year_color = year_color


def get_config(config_file: Optional[str] = None) -> Settings:
    """
    Get configuration settings, with defaults if config file is missing.
    
    Args:
        config_file (Optional[str]): Path to configuration file
        
    Returns:
        Settings: Configuration object with color settings
    """
    # Default configuration
    return Settings()
"""
Main application entry point for Hello World with year display.
"""

import datetime
from config import Settings, get_config


def display_hello_world_with_year(settings: Settings) -> None:
    """
    Display 'Hello World' with current year in specified colors.
    
    Args:
        settings (Settings): Configuration object with color settings
    """
    current_year = datetime.datetime.now().year
    text_color = settings.text_color
    year_color = settings.year_color
    
    # Create colored output using ANSI escape codes
    text_ansi = f"\033[{get_color_code(text_color)}mHello World\033[0m"
    year_ansi = f"\033[{get_color_code(year_color)}m{current_year}\033[0m"
    
    print(f"{text_ansi} {year_ansi}")


def get_color_code(color_name: str) -> str:
    """
    Get ANSI color code for a given color name.
    
    Args:
        color_name (str): Name of the color
        
    Returns:
        str: ANSI color code
    """
    colors = {
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37"
    }
    return colors.get(color_name, "37")  # Default to white


def main() -> None:
    """
    Main application function.
    """
    # Get configuration settings
    config = get_config()
    
    # Display the message
    display_hello_world_with_year(config)


if __name__ == "__main__":
    main()
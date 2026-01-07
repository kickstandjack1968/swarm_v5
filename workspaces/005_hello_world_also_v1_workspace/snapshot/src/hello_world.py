"""Hello World program with configurable text color and current year display."""

import json
import logging
from datetime import datetime
import os

# Mapping of supported colors to ANSI escape codes.
ANSI_CODES = {
    "red": "\033[31m",
    "white": "\033[37m",
    "blue": "\033[34m",
    "green": "\033[32m",
}

DEFAULT_COLOR = "red"
CONFIG_FILE = "config.json"
LOG_FILE = "app.log"


def load_config(path: str) -> str:
    """
    Load the color configuration from a JSON file.

    Returns the chosen color string. If the file is missing, malformed,
    or contains an unsupported color, the function falls back to
    DEFAULT_COLOR and logs the issue.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        color = data.get("color")
        if color not in ANSI_CODES:
            raise ValueError(f"Unsupported color: {color}")
        return color
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        logging.warning(
            f"Config error ({exc}). Using default color '{DEFAULT_COLOR}'."
        )
        return DEFAULT_COLOR


def setup_logging():
    """Configure file-based logging with timestamped entries."""
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Log each run
    logging.info("Program started.")


def display_message(color_name: str):
    """
    Print the current year centered above 'Hello, World!' in the chosen color.
    The terminal width is approximated; if unavailable, default to 80 columns.
    """
    # Determine terminal width
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    year_line = f"{datetime.now():%Y}"
    hello_line = "Hello, World!"

    # Center the year line
    centered_year = year_line.center(cols)
    print(centered_year)

    # Apply ANSI color to Hello World and reset after
    ansi_code = ANSI_CODES.get(color_name, ANSI_CODES[DEFAULT_COLOR])
    reset_code = "\033[0m"
    print(f"{ansi_code}{hello_line}{reset_code}")


def main():
    setup_logging()
    color = load_config(CONFIG_FILE)
    display_message(color)


if __name__ == "__main__":
    main()
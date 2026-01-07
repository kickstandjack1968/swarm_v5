```python
import json
from datetime import datetime

# Mapping of supported colors to ANSI escape codes.
ANSI_CODES = {
    "red": "\033[31m",
    "white": "\033[37m",
    "blue": "\033[34m",
    "green": "\033[32m",
}

DEFAULT_COLOR = "red"
CONFIG_FILE = "config.json"

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
    import logging
    logging.basicConfig(
        filename="app.log",
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
```

### Test Cases Using pytest

```python
import os
from datetime import datetime

import pytest
from .hello_world import load_config, display_message  # Adjust the import path as needed


# Fixture to mock terminal width for testing purposes.
@pytest.fixture(scope="function")
def mock_terminal_width(monkeypatch):
    monkeypatch.setattr("os.get_terminal_size", lambda: os.popen('stty size').read().splitlines()[0].split()[-1:])


def test_load_config_success(mock_terminal_width, caplog):
    # Test with valid config file
    caplog.clear()
    assert load_config(CONFIG_FILE) == "red"
    assert len(caplog.records) == 0


def test_load_config_missing_file(caplog):
    # Test missing config file
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.json")
    assert f"Config error (FileNotFoundError). Using default color '{DEFAULT_COLOR}'.\n" in caplog.text


def test_load_config_malformed_file(caplog):
    # Test malformed JSON file
    bad_config = '{"color": "invalid-color"}'
    with pytest.raises(json.JSONDecodeError):
        load_config("malformed.json")
    assert f"Config error (json.JSONDecodeError). Using default color '{DEFAULT_COLOR}'.\n" in caplog.text


def test_load_config_invalid_color(caplog):
    # Test invalid color name
    with open(CONFIG_FILE, "w") as f:
        f.write('{"color": "invalid"}')
    assert load_config(CONFIG_FILE) == "red"
    assert len(caplog.records) == 1 and caplog.text.startswith(f"Config error (ValueError). Using default color '{DEFAULT_COLOR}'.\n")


def test_display_message(mock_terminal_width):
    # Test display_message with valid configuration
    color_name = "green"
    display_message(color_name)
    output = os.popen("python hello_world.py").read()
    assert f"\033[32mHello, World!\033[0m" in output


def test_display_message_no_color(mock_terminal_width):
    # Test default color
    color_name = "nonexistent"
    display_message(color_name)
    output = os.popen("python hello_world.py").read()
    assert f"\033[31mHello, World!\033[0m" in output


def test_display_message_centered_year(mock_terminal_width):
    # Test year centered and aligned to the terminal width
    year_line = f"{datetime.now():%Y}"
    expected_output = f"{year_line.center(os.get_terminal_size().columns)}"
    assert display_message("red") == expected_output

# Run tests with pytest
if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
```

### Explanation of Test Cases:
- **Test Load Configuration**: Ensures the function successfully loads a valid configuration file.
- **Test Missing Config File**: Verifies that an exception is raised if the config file is missing, and it defaults to the default color with logging information.
- **Test Malformed Config File**: Checks for exceptions when loading malformed JSON files, ensuring they are handled gracefully by falling back to the default color and logging warnings.
- **Test Invalid Color Name**: Validates that invalid color names in the configuration result in the default color being used and proper error handling.
- **Test Display Message with Valid Configuration**: Ensures the `display_message` function correctly prints "Hello, World!" in a specified color and aligns it to center under the current year when available terminal width is mocked for testing purposes.

These tests should help ensure that your program handles various edge cases and configuration errors gracefully.
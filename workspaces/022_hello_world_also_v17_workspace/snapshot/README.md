# Hello World with Year

## Description
A small command‑line utility that prints a colored “Hello World” message along with the current year.

## Installation
1. Clone or download the repository and navigate to the source directory:
   ```bash
   cd src/
   ```
2. No external dependencies are required; the project uses only the Python standard library.

## Usage
```bash
python main.py
```
Running this command prints two pieces of text on the terminal:

* The current year (e.g., `2025`) in a color defined by `year_color`.
* The string “Hello World” in a color defined by `text_color`.

Both colors are represented as ANSI escape codes, so they will appear correctly on terminals that support ANSI coloring.

## API Reference

### `get_config(config_file: Optional[str] = None) -> Settings`
Retrieves configuration settings for text colors.  
- **Parameters**:
  - `config_file` (Optional[str]): Path to a configuration file (unused in the current implementation).  
- **Returns**: A `Settings` instance with default values (`text_color="white"`, `year_color="yellow"`).

### `display_hello_world_with_year(settings: Settings) -> None`
Prints “Hello World” and the current year using colors from a `Settings` object.  
- **Parameters**:
  - `settings` (Settings): Configuration containing `text_color` and `year_color`.  
- **Returns**: None.

### `get_color_code(color_name: str) -> str`
Converts a color name to its ANSI code.  
- **Parameters**:
  - `color_name` (str): Name of the color (e.g., `"red"`).  
- **Returns**: A string containing the ANSI numeric code; defaults to `"37"` (white) if the name is unknown or not provided.

### `main() -> None`
Entry point that obtains configuration via `get_config()` and calls `display_hello_world_with_year()`.

## Examples
Assuming the default settings:

* The year will appear in yellow.
* “Hello World” will appear in white.

If a user wants to change colors, they can instantiate a custom `Settings` object:

```python
from config import Settings

custom_settings = Settings(text_color="red", year_color="green")
display_hello_world_with_year(custom_settings)
```

Running the above would print the current year in green and “Hello World” in red.

## Edge Cases
* **Missing or malformed configuration file**: The `get_config` function ignores the `config_file` argument and always returns a default `Settings` instance, so no error is raised.
* **Invalid color names**: `get_color_code` falls back to white (`"37"`), ensuring that any unexpected input still results in colored output rather than an exception.

The implementation intentionally avoids external libraries for parsing configuration files or advanced terminal handling; all functionality relies on Python’s standard library and ANSI escape codes.
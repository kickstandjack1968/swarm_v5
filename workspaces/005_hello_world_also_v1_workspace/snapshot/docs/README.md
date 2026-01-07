# HelloWorldWithConfig

## Description
The `hello_world` project is designed as a simple terminal application that displays "Hello, World!" in the current year's format centered above it. The text can be colored according to an external configuration file. This program uses Python and supports configurable colors via JSON.

## Installation
To install and run this project, follow these steps:

1. Clone or download the repository.
2. Ensure you have a Python runtime environment installed on your machine (e.g., Python 3).
3. Create a `config.json` with at least one supported color in the format shown below:
   ```json
   {
       "color": "red"
   }
   ```
4. Run the program.

## Usage

1. Place the script and the configuration file (`config.json`) in the same directory.
2. Execute the script using Python:
   ```sh
   python hello_world.py
   ```

The terminal will display the current year centered above a colored "Hello, World!" line. Each execution of the program will append a timestamped entry to an `app.log` file.

## API Reference

### Functions

#### load_config(path: str) -> str
- **Description**: Loads the color configuration from a JSON file.
- **Parameters**:
  - `path`: The path to the configuration file. Must point to either `"config.json"` or another specified file if needed.
- **Returns**: A string representing the chosen color (e.g., "red", "white", etc.). If the file is missing, malformed, or contains an unsupported color, it falls back to `DEFAULT_COLOR` and logs a warning.

#### setup_logging()
- **Description**: Configures logging for each execution of the program.
- **Returns**: None

#### display_message(color_name: str)
- **Description**: Prints "Hello, World!" in the chosen color above the current year. The text is centered on the terminal line where possible; otherwise, it defaults to 80 columns.
- **Parameters**:
  - `color_name`: A string representing the desired text color from the ANSI codes supported by this application (e.g., "red", "white").
- **Returns**: None

#### main()
- **Description**: Coordinates all other functions: sets up logging, loads a configuration file to determine the color of the text, and displays the message.
- **Parameters**: None
- **Returns**: None

## Examples

To run the example:

1. Ensure `hello_world.py` and `config.json` are in the same directory.
2. Execute:
   ```sh
   python hello_world.py
   ```

You should see a terminal output similar to this (assuming "red" is chosen as the default color):
```
2023
Hello, World!
```

If you modify `config.json` and run again:

```json
{
    "color": "blue"
}
```

The output will be:
```
2023
Hello, World!
```

Each execution will also log a message to the `app.log`.

## Dependencies

This project uses only Python's built-in libraries and does not require any external dependencies.
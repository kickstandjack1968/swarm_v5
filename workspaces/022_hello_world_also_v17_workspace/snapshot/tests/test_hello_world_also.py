from unittest.mock import patch, mock_open
import pytest
from main import display_hello_world_with_year, get_color_code, main, Settings
from config import get_config


def test_get_color_code_valid_colors():
    """Test that valid color names return correct ANSI codes."""
    assert get_color_code("black") == "30"
    assert get_color_code("red") == "31"
    assert get_color_code("green") == "32"
    assert get_color_code("yellow") == "33"
    assert get_color_code("blue") == "34"
    assert get_color_code("magenta") == "35"
    assert get_color_code("cyan") == "36"
    assert get_color_code("white") == "37"


def test_get_color_code_invalid_color():
    """Test that invalid color names default to white."""
    assert get_color_code("invalid") == "37"
    assert get_color_code("") == "37"


def test_display_hello_world_with_year():
    """Test display function with custom settings."""
    settings = Settings(text_color="red", year_color="green")
    
    with patch('sys.stdout') as mock_stdout:
        display_hello_world_with_year(settings)
        mock_stdout.write.assert_called()


def test_main_function():
    """Test main function execution."""
    with patch('main.get_config') as mock_get_config:
        mock_settings = Settings(text_color="blue", year_color="yellow")
        mock_get_config.return_value = mock_settings
        
        with patch('main.display_hello_world_with_year') as mock_display:
            main()
            mock_display.assert_called_once_with(mock_settings)


def test_main_function_default_config():
    """Test main function with default configuration."""
    with patch('main.get_config') as mock_get_config:
        mock_settings = Settings()
        mock_get_config.return_value = mock_settings
        
        with patch('main.display_hello_world_with_year') as mock_display:
            main()
            mock_display.assert_called_once_with(mock_settings)


def test_get_config_default():
    """Test that get_config returns default Settings when no file provided."""
    result = get_config()
    assert isinstance(result, Settings)
    assert result.text_color == "white"
    assert result.year_color == "yellow"


def test_get_color_code_edge_cases():
    """Test edge cases for color code function."""
    # Test None input
    assert get_color_code(None) == "37"
    
    # Test case sensitivity
    assert get_color_code("RED") == "37"
    assert get_color_code("Red") == "37"
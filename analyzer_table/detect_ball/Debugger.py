"""
A simple, color-coded logger for debugging the analysis pipeline.

This module provides a static 'Debugger' class that allows for logging
messages with different severity levels (DEBUG, WARN, ERROR), each displayed
in a distinct color. The logging can be globally enabled or disabled by setting
the DEBUG_MODE flag.

Note:
    This is a basic implementation and is not a substitute for a full-featured
    logging library like Python's `logging` module. It lacks runtime
    configurability, log levels, and different output handlers.
"""

# Global flag to enable or disable debug logging across the application.
DEBUG_MODE = True


class Debugger:
    """
    A static class providing color-coded logging methods for console output.
    """

    # ANSI escape codes for terminal colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    @staticmethod
    def log(message: str) -> None:
        """
        Prints a green [DEBUG] message to the console if DEBUG_MODE is True.

        Args:
            message: The message to be logged.
        """
        if DEBUG_MODE:
            print(f"{Debugger.GREEN}[DEBUG]{Debugger.RESET} {message}")

    @staticmethod
    def warn(message: str) -> None:
        """
        Prints a yellow [WARN] message to the console if DEBUG_MODE is True.

        Args:
            message: The warning message to be logged.
        """
        if DEBUG_MODE:
            print(f"{Debugger.YELLOW}[WARN]{Debugger.RESET} {message}")

    @staticmethod
    def error(message: str) -> None:
        """
        Prints a red [ERROR] message to the console if DEBUG_MODE is True.

        Args:
            message: The error message to be logged.
        """
        if DEBUG_MODE:
            print(f"{Debugger.RED}[ERROR]{Debugger.RESET} {message}")

class APIError(Exception):
    """Custom exception class for handling API-related errors."""

    def __init__(self, message: str):
        """Initialize the API error.

        Args:
            message (str): Human-readable error message
        """
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a string representation of the error."""
        return self.message

"""
I/O utilities for file handling and output redirection.
"""

import os
import sys
from datetime import datetime
from typing import Optional, Tuple


class TeeOutput:
    """Write to both file and original stream (stdout/stderr)."""

    def __init__(self, file_path: str, original_stream):
        self.file = open(file_path, "a", buffering=1)  # Line buffered
        self.original = original_stream

    def write(self, data):
        self.file.write(data)
        self.original.write(data)

    def flush(self):
        self.file.flush()
        self.original.flush()

    def close(self):
        self.file.close()


def setup_output_logging(
    log_file: Optional[str],
) -> Tuple[Optional[TeeOutput], Optional[TeeOutput]]:
    """
    Redirect stdout and stderr to both console and file.

    Args:
        log_file: Path to log file. If None, no redirection occurs.

    Returns:
        Tuple of (stdout_tee, stderr_tee). Both None if log_file is None.
    """
    if log_file is None:
        return None, None

    # Create log directory if needed
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Add timestamp header to log file
    with open(log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(
            f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"{'='*80}\n\n")

    # Redirect stdout and stderr
    stdout_tee = TeeOutput(log_file, sys.stdout)
    stderr_tee = TeeOutput(log_file, sys.stderr)

    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    return stdout_tee, stderr_tee

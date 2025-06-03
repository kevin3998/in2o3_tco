"""
Module: logging_config
Functionality: Configures basic logging settings for the application.
               This includes setting the logging level, format, date format,
               and handlers (e.g., stream to console, file).
"""

import logging
import sys

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
            # You can add logging.FileHandler("extraction.log") here
        ]
    )
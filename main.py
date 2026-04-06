from __future__ import annotations

import logging

from src.utils.logging import setup_logging


def main() -> None:
    setup_logging(
        app_name="lslm",
    )
    logging.info("LSLM started.")
if __name__ == "__main__":
    main()
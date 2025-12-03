import logging, os
from colorama import init, Fore, Style

init(autoreset=True)

class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.CRITICAL: Fore.LIGHTYELLOW_EX + Style.BRIGHT,  # Orange-like
        logging.ERROR: Fore.RED + Style.BRIGHT,
        logging.WARNING: Fore.LIGHTRED_EX,
        logging.INFO: Fore.WHITE,
        logging.DEBUG: Fore.CYAN,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = color + record.levelname + Style.RESET_ALL
        record.msg = color + record.message + Style.RESET_ALL
        return super().format(record)


def setup_logging(log_path, delete_existing=True, level=logging.DEBUG):
    """
    Set up a logger that logs to both console and a file.
    Critical -> Orange, Warning -> Light Red (console only)
    """
    if delete_existing and os.path.exists(log_path):
        os.remove(log_path)

    logger = logging.getLogger("train.hlp")
    logger.setLevel(level)

    if not logger.handlers:
        # Base format (no color codes here; apply them in ColorFormatter for console)
        base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        # File handler (no color)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(logging.Formatter(base_format, datefmt=date_format))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

        # Console handler (with color)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColorFormatter(base_format, datefmt=date_format))
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

def silence_logger(logger: logging.Logger):
    # remove any existing handlers (file/console) to avoid side effects
    logger.handlers.clear()
    # prevent bubbling up to root logger (which might have handlers)
    logger.propagate = False

    logger.setLevel(logging.CRITICAL + 1)
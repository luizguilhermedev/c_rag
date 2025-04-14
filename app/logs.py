import logging

from app.settings import settings

loggers = {}


def get_logger(name: str = "clean_rag") -> logging.Logger:
    """
    Creates a default python logger configured for the app
    """
    if name in loggers:
        return loggers[name]
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOGGING_LEVEL))
    formatter = logging.Formatter(
        "{asctime} {threadName:>11} {levelname} {filename} {message}", style="{"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, settings.LOGGING_LEVEL))
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if settings.LOGGING_FILE:
        file_handler = logging.FileHandler(
            settings.LOGGING_FILE,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, settings.LOGGING_LEVEL))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    loggers[name] = logger
    return logger

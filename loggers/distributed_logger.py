from base_logger import BaseLogger


class DistributedLogger:
    """
    Wrapper around a logger to make it thread-safe. Only the main process will log messages, while the other processes will do nothing.
    """

    def __init__(self, logger, accelerator):
        self.logger = logger
        self.accelerator = accelerator

    def __getattr__(self, name):
        # get the actual attribute from the real logger
        attr = getattr(self.logger, name)

        # if it's a method, wrap it in a function that checks if we're on the main process before calling it
        if callable(attr):

            def wrapper(*args, **kwargs):
                if self.accelerator.is_main_process and self.logger:
                    return attr(*args, **kwargs)
                return None

            return wrapper

        return attr

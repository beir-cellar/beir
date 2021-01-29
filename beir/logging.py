import logging
import sys
from logging import Logger


class LoggingHandler(Logger):
    def __init__(
        self,
        level=logging.INFO, 
        log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        *args,
        **kwargs
    ):
        self.formatter = logging.Formatter(log_format)
        self.level = level

        Logger.__init__(self, *args, **kwargs)

        self.addHandler(self.get_console_handler())
        # with this pattern, it's rarely necessary to propagate the| error up to parent
        self.propagate = False

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.formatter)
        return console_handler

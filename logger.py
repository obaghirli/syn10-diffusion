import os
import sys
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Union
from globals import Globals


class TBSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: Union[str, Path]):
        if not isinstance(log_dir, Path):
            log_dir = Path(log_dir)
        self.log_dir = log_dir
        super().__init__(log_dir=str(log_dir))


class DistributedLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        rank = int(os.environ['RANK'])
        if rank == 0 and not hasattr(self._instance, 'logger'):
            self.rank = rank
            self.params = Globals().params
            self.is_train = self.params['is_train']
            self.run_dir = Path(self.params['artifact_dir']) / self.params['run_id']
            self.log_file = self.run_dir / 'log_train.txt' if self.is_train else self.run_dir / 'log_val.txt'
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)

            # Create a file handler and set its logging level
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)

            # Create a stream handler to log to stdout and set its logging level
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)

            # Create a log format
            log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(log_format)
            stdout_handler.setFormatter(log_format)

            # Add the file handler and stdout handler to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stdout_handler)

    def log_debug(self, message):
        if self.rank == 0:
            self.logger.debug(message)

    def log_info(self, message):
        if self.rank == 0:
            self.logger.info(message)

    def log_warning(self, message):
        if self.rank == 0:
            self.logger.warning(message)

    def log_error(self, message):
        if self.rank == 0:
            self.logger.error(message)

    def log_critical(self, message):
        if self.rank == 0:
            self.logger.critical(message)


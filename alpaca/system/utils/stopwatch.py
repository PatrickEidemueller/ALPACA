from datetime import datetime, timedelta
import traceback


class StopWatch(object):
    """
    StopWatch class that can be used like 'with StopWatch() as stopwatch: ...'
    """
    def __enter__(self) -> "StopWatch":
        """
        Starts the stopwatch
        """
        self.start = datetime.now()
        return self

    def elapsed(self) -> timedelta:
        """
        Returns the timedelta since the stopwatch was started
        """
        return datetime.now() - self.start

    def __exit__(self, exc_type, exc_value, tb):
        """
        When exiting context prints traceback
        """
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False
        return True

import logging
import sys
from collections.abc import Callable

from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


class InitWrapper[**P, R]:
    """Initialization wrapper for Hydra-Zen that catches exceptions and logs them.

    Can optionally be combined with further wrappers, e.g. for pydantic validation.
    """

    def __init__(
        self,
        wrapper: Callable[[Callable[P, R]], Callable[P, R]] | None = None,
        **wrapper_kwargs,
    ):
        self.wrapper = wrapper  # Instantiation wrapper, e.g. pydantic
        self.wrapper_kwargs = wrapper_kwargs
        logging.captureWarnings(True)  # Log Python-native warnings

    def __call__(self, target: Callable[P, R]) -> Callable[P, R]:
        func = self.wrapper(target, **self.wrapper_kwargs) if self.wrapper else target

        def exception_logger(*args: P.args, **kwargs: P.kwargs) -> R:
            """Wraps around the initialized functions and catches exceptions."""
            try:
                with logging_redirect_tqdm():
                    return func(*args, **kwargs)
            except KeyboardInterrupt:
                logger.error("Interrupted by user.")
                sys.exit(130)  # Standard exit code for KeyboardInterrupt
            except Exception as exc:
                message = ". ".join([ll.strip(". ") for ll in str(exc).split("\n")])
                logger.critical(message, exc_info=True)
                sys.exit(1)

        return exception_logger

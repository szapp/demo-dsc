from typing import overload

import pandas as pd
import pandera.pandas as pa


@overload
def is_int(p: pd.DataFrame) -> pd.DataFrame: ...


@overload
def is_int(p: pd.Series) -> pd.Series: ...


def is_int(p):
    """Check float input if integer or NaN to allow 'integer-floats'."""
    return (p % 1 == 0) | p.isna()


# Only new checks register once
if not hasattr(pa.Check, "is_int"):  # pragma: no cover
    pa.extensions.register_check_method(is_int)

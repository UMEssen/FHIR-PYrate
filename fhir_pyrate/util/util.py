import datetime
from typing import Any

import pandas as pd


def string_from_column(
    col: pd.Series,
    separator: str = ", ",
    unique: bool = False,
    sort: bool = False,
    sort_reverse: bool = False,
) -> Any:
    """
    Transforms the values contained in a pandas Series into a string of (if desired unique) values.

    :param col:
    :param separator: The separator for the values
    :param unique: Whether only unique values should be stored
    :param sort: Whether the values should be sorted
    :param sort_reverse: Whether the values should sorted in reverse order
    :return: A string containing the values of the Series.
    """
    existing_values = list()
    for el in col.values:
        if not pd.isnull(el) and el != "":
            existing_values.append(el)
    if unique:
        existing_values = list(set(existing_values))
    if len(existing_values) == 0:
        return None
    elif len(existing_values) == 1:
        return existing_values.pop()
    else:
        values = [str(el) for el in existing_values]
    if sort:
        values.sort(reverse=sort_reverse)
    return separator.join(values)


def get_datetime(dt_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Creates a datetime string according to the given format

    :param dt_format: The format to use for the printing
    :return: The formatted string
    """
    return datetime.datetime.now().strftime(dt_format)

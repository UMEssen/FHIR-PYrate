import re
from importlib import import_module
from typing import Any, Tuple

PACKAGE_TO_CLASS = {
    "spacy": "miner",
    "pydicom": "downloader",
    "SimpleITK": "downloader",
    "dicomweb_client": "downloader",
}


"""
The following code has been taken from MONAI
https://github.com/Project-MONAI/MONAI/blob/ed233d9b48bd71eb623cf9777ad9b60142c8ad66/monai/utils/module.py
and modified as follows:
* The function `optional_import` does not take a version parameter.
* The format of the `optional_import` docstring has been changed to reStructuredText.
* The `_default_msg` parameter contains information that are related to this package.
* Addition of type hints to `_LazyRaise`.

MONAI is licenced under Apache License 2.0.
A copy of the license text can be found at
https://github.com/Project-MONAI/MONAI/blob/ed233d9b48bd71eb623cf9777ad9b60142c8ad66/LICENSE
"""

OPTIONAL_IMPORT_MSG_FMT = "{}"


class OptionalImportError(ImportError):
    """
    Could not import APIs from an optional dependency.
    """


def optional_import(
    module: str,
    name: str = "",
    descriptor: str = OPTIONAL_IMPORT_MSG_FMT,
    allow_namespace_pkg: bool = False,
) -> Tuple[Any, bool]:
    """
    Imports an optional module specified by `module` string.
    Any importing related exceptions will be stored, and exceptions raise lazily
    when attempting to use the failed-to-import module.

    :param module: name of the module to be imported.
    :param name: a non-module attribute (such as method/class) to import from the imported module.
    :param descriptor: a format string for the final error message when using a not imported module.
    :param allow_namespace_pkg: whether importing a namespace package is allowed. Defaults to False.
    :return: The imported module and a boolean flag indicating whether the import is successful.
    """
    tb = None
    if name:
        actual_cmd = f"from {module} import {name}"
    else:
        actual_cmd = f"import {module}"
    try:
        the_module = import_module(module)
        if not allow_namespace_pkg:
            is_namespace = getattr(the_module, "__file__", None) is None and hasattr(
                the_module, "__path__"
            )
            if is_namespace:
                raise AssertionError
        if name:  # user specified to load class/function/... from the module
            the_module = getattr(the_module, name)
    except Exception as import_exception:  # any exceptions during import
        tb = import_exception.__traceback__
        exception_str = f"{import_exception}"
    else:  # found the module
        return the_module, True

    # preparing lazy error message
    msg = descriptor.format(actual_cmd)
    if exception_str:
        msg += f" ({exception_str})"

    class _LazyRaise:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            match = re.match(pattern=r"No module named '([\w]+)'", string=exception_str)
            if match is not None:
                missing_package = match.group(1)
                package_string = f"The package {missing_package} is not installed."
                if missing_package in PACKAGE_TO_CLASS:
                    package_string += (
                        f" Please install it with pip install "
                        f"'fhir-pyrate[{PACKAGE_TO_CLASS[missing_package]}]' "
                        f"(or 'fhir-pyrate[all]') or add the optional dependency "
                        f"`{PACKAGE_TO_CLASS[missing_package]}` "
                        "(or `all`) to your package manager."
                    )
            else:
                package_string = ""
            _default_msg = f"{msg}.\n\n{package_string}"
            if tb is None:
                self._exception = OptionalImportError(_default_msg)
            else:
                self._exception = OptionalImportError(_default_msg).with_traceback(tb)

        def __getattr__(self, name: str) -> str:
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

        def __call__(self, *_args: Any, **_kwargs: Any) -> str:
            """
            Raises:
                OptionalImportError: When you call this method.
            """
            raise self._exception

    return _LazyRaise(), False

import importlib.util
import logging

from fhir_pyrate.ahoy import Ahoy
from fhir_pyrate.pirate import Pirate

logging.basicConfig()
logger = logging.getLogger()

DICOM_PACKAGES = ["dicomweb_client", "pydicom", "SimpleITK"]

if importlib.util.find_spec("spacy") is None:
    logger.warning(
        "The package spacy is not installed, so you will not be able to use the "
        "Miner. "
        "Please install them with pip install fhir-pyrate[miner] "
        "(or fhir-pyrate[all]) or add the optional dependency `miner` "
        "(or `all`) to your package manager."
    )
else:
    from fhir_pyrate.miner import Miner

not_installed = [
    importlib.util.find_spec(package) is None for package in DICOM_PACKAGES
]
if any(not_installed):
    to_install = [package for b, package in zip(not_installed, DICOM_PACKAGES) if b]
    logger.warning(
        f"The packages {to_install} are not installed, so you will not be able to use the "
        "DicomDownloader. "
        "Please install them with pip install fhir-pyrate[downloader] "
        "(or fhir-pyrate[all]) or add the optional dependency `downloader` "
        "(or `all`) to your package manager."
    )
else:
    from fhir_pyrate.dicom_downloader import DicomDownloader

__all__ = [
    "Ahoy",
    "Miner",
    "Pirate",
    "DicomDownloader",
]

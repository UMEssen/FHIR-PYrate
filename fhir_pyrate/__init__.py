import importlib.util
import logging

from fhir_pyrate.ahoy import Ahoy
from fhir_pyrate.pirate import Pirate

DICOM_PACKAGES = ["dicomweb_client", "pydicom", "SimpleITK"]

if importlib.util.find_spec("spacy") is None:
    logging.warning(
        "SpaCy is not installed, so you will not be able to use the Miner."
        "Please install spacy with pip install fhir-pyrate[miner] or add the "
        "optional dependency `miner` to your package manager."
    )
else:
    from fhir_pyrate.miner import Miner

not_installed = [
    importlib.util.find_spec(package) is None for package in DICOM_PACKAGES
]
if any(not_installed):
    to_install = [package for b, package in zip(not_installed, DICOM_PACKAGES) if b]
    logging.warning(
        f"The packages {to_install} are not installed, so you will not be able to use the "
        "DICOMDownloader."
        "Please install them with pip install fhir-pyrate[downloader] or add the "
        "optional dependency `downloader` to your package manager."
    )
else:
    from fhir_pyrate.dicom_downloader import DicomDownloader

__all__ = [
    "Ahoy",
    "Miner",
    "Pirate",
    "DicomDownloader",
]

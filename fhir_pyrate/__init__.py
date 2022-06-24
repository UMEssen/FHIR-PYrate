import logging

import fhir_pyrate.util.imports
from fhir_pyrate.ahoy import Ahoy
from fhir_pyrate.pirate import Pirate

logging.basicConfig()
logger = logging.getLogger(__name__)

Miner, _ = fhir_pyrate.util.imports.optional_import(
    module="fhir_pyrate.miner", name="Miner"
)
DicomDownloader, _ = fhir_pyrate.util.imports.optional_import(
    module="fhir_pyrate.dicom_downloader", name="DicomDownloader"
)

__all__ = [
    "Ahoy",
    "Miner",
    "Pirate",
    "DicomDownloader",
]

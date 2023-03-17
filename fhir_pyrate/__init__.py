import importlib.metadata
import logging

import fhir_pyrate.util.imports
from fhir_pyrate.ahoy import Ahoy
from fhir_pyrate.pirate import Pirate

__version__ = importlib.metadata.version("fhir_pyrate")
logging.basicConfig()
logging.captureWarnings(True)
# warnings.warn() in library code if the issue is avoidable and the client application
# should be modified to eliminate the warning

# logging.warning() if there is nothing the client application can do about the situation,
# but the event should still be noted

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

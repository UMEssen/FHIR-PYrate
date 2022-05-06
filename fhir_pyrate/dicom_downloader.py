import datetime
import hashlib
import io
import logging
import os
import pathlib
import platform
import shutil
import sys
import tempfile
import traceback
import warnings
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Dict, Generator, List, Optional, TextIO, Tuple, Type, Union

import pandas as pd
import pydicom
import requests
import SimpleITK as sitk
from dicomweb_client.api import DICOMwebClient
from tqdm import tqdm

from fhir_pyrate.util import get_datetime

##########
# Follows a workaround to detect SimpleITK warnings
# Thank you kind stranger from the internet
# https://stackoverflow.com/questions/64674495/how-to-catch-simpleitk-warnings
# https://github.com/hankcs/HanLP/blob/doc-zh/hanlp/utils/io_util.py
try:
    import ctypes
    from ctypes.util import find_library
except ImportError:
    libc = None
else:
    try:
        libc = ctypes.cdll.msvcrt  # Windows
    except OSError:
        libc = ctypes.cdll.LoadLibrary(find_library("c"))  # type: ignore


def flush(stream: TextIO) -> None:
    try:
        assert libc is not None
        libc.fflush(None)
        stream.flush()
    except (AttributeError, ValueError, IOError, AssertionError):
        pass  # unsupported


def fileno(file_or_fd: TextIO) -> Optional[int]:
    try:
        fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    except io.UnsupportedOperation:
        return None
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(
    to: Union[str, TextIO] = os.devnull, stdout: Optional[TextIO] = None
) -> Generator:
    if platform.system() == "Windows":
        yield None
        return
    if stdout is None:
        stdout = sys.stdout
    stdout_fd = fileno(stdout)
    if not stdout_fd:
        yield None
        return
    # copy stdout_fd before it is overwritten
    # Note: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        # stdout.flush()  # flush library buffers that dup2 knows nothing about
        # stdout.flush() does not flush C stdio buffers on Python 3 where I/O is
        # implemented directly on read()/write() system calls. To flush all open C stdio
        # output streams, you could call libc.fflush(None) explicitly if some C extension uses
        # stdio-based I/O:
        flush(stdout)
        try:
            # $ exec >&to
            os.dup2(fileno(to), stdout_fd)  # type: ignore
        except ValueError:  # filename
            with open(to, "wb") as to_file:  # type: ignore
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # Note: dup2 makes stdout_fd inheritable unconditionally
            # stdout.flush()
            flush(stdout)
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


##########


class DicomDownloader:
    """
    Downloads DICOM series and studies and stores their mapping to a patient in a CSV file.
    :param auth: The authentication to the server to download the files
    :param dicom_web_url: The DicomWeb URL used to download the files
    :param output_format: The options are [nifti, DICOM]:
    DICOM will leave the files as they are;
    nifti will convert them
    :param retry:
    """

    def __init__(
        self,
        auth: Any,
        dicom_web_url: str,
        output_format: str = "nifti",
        retry: bool = False,
    ):
        self.auth = auth
        self.dicom_web_url = dicom_web_url
        if self.auth is not None:
            self.session = self.auth.session
        else:
            self.session = requests.session()
        self.client = self._init_dicom_web_client()
        self.client.set_http_retry_params(retry=retry)
        if output_format.lower() not in ["nifti", "dicom"]:
            raise ValueError(f"The given format {output_format} is not supported.")
        self._output_format = output_format.lower()
        self.study_instance_uid_field = "study_instance_uid"
        self.series_instance_uid_field = "series_instance_uid"
        self.deid_study_instance_uid_field = "deidentified_study_instance_uid"
        self.deid_series_instance_uid_field = "deidentified_series_instance_uid"
        self.download_id_field = "download_id"
        self.error_type_field = "error"
        self.traceback_field = "traceback"

    def set_output_format(self, new_output_format: str) -> None:
        """
        Change the output format of the class

        :param new_output_format: The new output format
        :return: None
        """
        if new_output_format.lower() not in ["nifti", "dicom"]:
            raise ValueError(f"The given format {new_output_format} is not supported.")
        self._output_format = new_output_format.lower()

    def __enter__(self) -> "DicomDownloader":
        return self

    def _init_dicom_web_client(self) -> DICOMwebClient:
        """
        Init the DicomWebClient.

        :return: None
        """
        client = DICOMwebClient(
            session=self.session,
            url=self.dicom_web_url,
        )
        return client

    def close(self) -> None:
        # Only close the session if it does not come from an authentication class
        if self.auth is None:
            self.session.close()

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        self.close()

    @staticmethod
    def get_download_id(study_uid: str, series_uid: str = None) -> str:
        """
        Generate a download id given the study and (if available) the series ID.

        :param study_uid: The study ID to download
        :param series_uid: The series ID to download
        :return: An encoded combination of study and series ID
        """
        key_string = study_uid
        if series_uid is not None:
            key_string += "_" + series_uid
        return hashlib.sha256(key_string.encode()).hexdigest()

    def download_data(
        self,
        study_uid: str,
        series_uid: str = None,
        output_dir: Union[str, pathlib.Path] = "out",
        save_metadata: bool = True,
        existing_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Downloads the data related to the StudyInstanceUID and SeriesInstanceUID (if given,
        otherwise the entire study will be downloaded).

        :param study_uid: The StudyInstanceUID
        :param series_uid: The SeriesInstanceUID
        :param output_dir: The output directory where the files should be stored
        :param save_metadata: Whether to store the metadata
        :param existing_ids: A list of download IDs that have already been previously downloaded
        :return: Two list of dictionaries: First, the successfully downloaded download IDs together
        with some additional information, such as StudyInstanceUID, SeriesInstanceUID and
        identified IDs; Second, the studies that have failed to download together with some
        additional information such as the type of error and the traceback
        """
        output_dir = pathlib.Path(output_dir)
        downloaded_series_info: List[Dict[str, str]] = []
        error_series_info: List[Dict[str, str]] = []
        # Generate a hash of key/series which will be the ID of this download
        download_id = self.get_download_id(study_uid=study_uid, series_uid=series_uid)
        # Recompute if the hash does not exist in existing IDs, in case existing IDs is given
        recompute = (
            download_id not in existing_ids if existing_ids is not None else False
        )
        file_format = ".nii.gz" if self._output_format == "nifti" else ".dcm"
        logging.info(f"{get_datetime()} Current download ID: {download_id}")
        # Check if it exists already and skips
        if (
            len(list((output_dir / download_id).glob(f"*{file_format}"))) > 0
            and not recompute
        ):
            logging.info(
                f"Study {download_id} has been already downloaded in "
                f"{output_dir}, skipping..."
            )
            return downloaded_series_info, error_series_info

        base_dict = {
            self.study_instance_uid_field: study_uid,
            self.download_id_field: download_id,
        }
        if series_uid is not None:
            base_dict[self.series_instance_uid_field] = series_uid

        # Init the readers/writers
        series_reader = sitk.ImageSeriesReader()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create the download dir
            current_tmp_dir = pathlib.Path(tmp_dir)
            progress_bar = tqdm(leave=False, desc="Downloading")
            # Save the DICOMs to the tmpdir
            try:
                it = (
                    self.client.iter_study(study_uid)
                    if series_uid is None
                    else self.client.iter_series(study_uid, series_uid)
                )
                for dcm in it:
                    dcm.save_as(current_tmp_dir / f"{dcm.SOPInstanceUID}.dcm")
                    progress_bar.update()
            except (
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.HTTPError,
            ):
                logging.debug(traceback.format_exc())
                progress_bar.close()
                logging.info(f"Study {download_id} could not be fully downloaded.")
                base_dict[self.error_type_field] = "Download Error"
                base_dict[self.traceback_field] = traceback.format_exc()
                return [], [base_dict]
            progress_bar.close()

            # Get Series ID names from folder
            series_uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(current_tmp_dir))
            logging.info(f"Study ID has {len(series_uids)} series.")
            for series in series_uids:
                # Get the DICOMs corresponding to the series
                files = series_reader.GetGDCMSeriesFileNames(
                    str(current_tmp_dir), series
                )
                current_dict = base_dict.copy()
                simpleitk_warning_file = (
                    current_tmp_dir / f"{series}_warning_output.txt"
                )
                try:
                    # Read the series
                    with simpleitk_warning_file.open("w") as f, stdout_redirected(
                        f, stdout=sys.stderr
                    ):
                        series_reader.SetFileNames(files)
                        image = series_reader.Execute()
                    with simpleitk_warning_file.open("r") as f:
                        content = f.read()
                    if "warning" in content.lower():
                        raise RuntimeError("SimpleITK " + content)
                    # Create the final output dir
                    (output_dir / download_id).mkdir(exist_ok=True, parents=True)
                    if self._output_format == "nifti":
                        # Write series to nifti
                        sitk.WriteImage(
                            image, str(output_dir / download_id / (series + ".nii.gz"))
                        )
                    else:
                        # We just copy the dicoms
                        for dcm_file in files:
                            shutil.copy2(
                                dcm_file,
                                output_dir
                                / download_id
                                / f"{pathlib.Path(dcm_file).name}",
                            )
                except Exception:
                    logging.debug(traceback.format_exc())
                    logging.info(f"Series {series} could not be stored.")
                    current_dict[self.error_type_field] = "Storing Error"
                    current_dict[self.traceback_field] = traceback.format_exc()
                    error_series_info.append(current_dict)
                    continue

                # Store one DICOM to keep the file metadata
                if save_metadata and self._output_format == "nifti":
                    shutil.copy2(
                        files[0],
                        output_dir / download_id / f"{series}_meta.dcm",
                    )
                dcm_info = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
                current_dict[
                    self.deid_study_instance_uid_field
                ] = dcm_info.StudyInstanceUID
                current_dict[self.deid_series_instance_uid_field] = series
                downloaded_series_info.append(current_dict)

        return downloaded_series_info, error_series_info

    def fix_mapping_dataframe(
        self,
        df: pd.DataFrame,
        mapping_df: pd.DataFrame = None,
        output_dir: Union[str, pathlib.Path] = "out",
        study_uid_col: str = "study_instance_uid",
        series_uid_col: str = "series_instance_uid",
        skip_existing: bool = True,
    ) -> pd.DataFrame:
        """
        Go through the already downloaded data and check if there are some instances that have
        not been stored in the mapping file. A new mapping file with the given name will be stored.

        :param df: The DataFrame that contains the studies that should be downloaded
        :param mapping_df: A DataFrame containing the generated mappings until now
        :param output_dir: The directory where the studies are stored
        :param study_uid_col: The name of the StudyInstanceUID column of the DataFrame
        :param series_uid_col: The name of the SeriesInstanceUID column of the DataFrame
        :param skip_existing: Whether existing studies should be skipped
        :return: The fixed mapping DataFrame
        """
        output_dir = pathlib.Path(output_dir)
        assert (
            output_dir.exists()
        ), "Cannot fix the mapping file if the output directory does not exist."
        if mapping_df is None:
            mapping_df = pd.DataFrame()
        csv_rows = []
        for row in df.itertuples(index=False):
            # Create download id
            download_id = DicomDownloader.get_download_id(
                study_uid=getattr(row, study_uid_col),
                series_uid=getattr(row, series_uid_col),
            )
            # If the files have not been downloaded or
            # if the ID is already in the mapping dataframe
            if (
                len(mapping_df) > 0
                and skip_existing
                and download_id in mapping_df[self.download_id_field].values
            ) or len(list((output_dir / download_id).glob("*"))) == 0:
                continue
            base_dict = {
                self.study_instance_uid_field: getattr(row, study_uid_col),
                self.download_id_field: download_id,
            }
            if getattr(row, series_uid_col) is not None:
                base_dict[self.series_instance_uid_field] = getattr(row, series_uid_col)
            # Collect the pairs of study instance ID and series instance ID that exist
            existing_pairs = set()
            for dcm_file in (output_dir / download_id).glob("*.dcm"):
                dcm_info = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                existing_pairs.add(
                    (dcm_info.StudyInstanceUID, dcm_info.SeriesInstanceUID)
                )
            for destudy, deseries in existing_pairs:
                current_dict = base_dict.copy()
                current_dict[self.deid_study_instance_uid_field] = destudy
                current_dict[self.deid_series_instance_uid_field] = deseries
                csv_rows.append(current_dict)
        logging.info(f"{len(csv_rows)} have been fixed.")
        new_df = pd.concat([mapping_df, pd.DataFrame(csv_rows)])
        return new_df

    def download_data_from_dataframe(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, pathlib.Path] = "out",
        study_uid_col: str = "study_instance_uid",
        series_uid_col: Optional[str] = "series_instance_uid",
        mapping_df: pd.DataFrame = None,
        download_full_study: bool = False,
        save_metadata: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :param df: The DataFrame that contains the studies that should be downloaded
        :param output_dir: The directory where the studies are stored
        :param study_uid_col: The name of the StudyInstanceUID column of the DataFrame
        :param series_uid_col: The name of the SeriesInstanceUID column of the DataFrame
        :param mapping_df: A mapping DataFrame that contains the mapping between the
        StudyInstanceUID/SeriesInstanceUID and the download IDs
        :param download_full_study: Whether the full study should be downloaded or only the series
        :param save_metadata: Whether to save some metadata: In case the output is a DICOM,
        the metadata is always stored, otherwise the first DICOM file will be copied to the
        output folder
        :return: Two DataFrames: The first one contains the IDs of all the successfully
        downloaded studies, while the second ones has the IDs of the studies that failed,
        the kind of error, and the traceback
        """
        # Create the mapping file
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if mapping_df is None:
            mapping_df = pd.DataFrame(
                columns=[
                    self.study_instance_uid_field,
                    self.series_instance_uid_field,
                    self.deid_study_instance_uid_field,
                    self.deid_series_instance_uid_field,
                    self.download_id_field,
                ]
            )
        existing_ids = mapping_df[self.download_id_field].values
        if study_uid_col not in df.columns:
            raise ValueError(
                "The given Study ID Column is not a valid column of the DataFrame."
            )
        if not download_full_study and (
            series_uid_col is None or series_uid_col not in df.columns
        ):
            download_full_study = True
            warnings.warn(
                "download_full_study = False will only download a specified series but "
                "have not provided a valid Series UID column of the DataFrame, "
                "as a result the full study will be downloaded."
            )

        # Create list of rows
        csv_rows = []
        error_rows = []
        for row in df.itertuples(index=False):
            try:
                download_info, error_info = self.download_data(
                    study_uid=getattr(row, study_uid_col),
                    series_uid=getattr(row, series_uid_col)
                    if not download_full_study and series_uid_col is not None
                    else None,
                    output_dir=output_dir,
                    save_metadata=save_metadata,
                    existing_ids=existing_ids,
                )
            except KeyboardInterrupt:
                break
            except Exception:
                # If any error happens that is not caught, just go to the next one
                error_rows += [
                    {
                        self.study_instance_uid_field: getattr(row, study_uid_col),
                        self.series_instance_uid_field: getattr(row, series_uid_col)
                        if isinstance(series_uid_col, str)
                        and getattr(row, series_uid_col) is not None
                        else None,
                        self.error_type_field: "Other Error",
                        self.traceback_field: traceback.format_exc(),
                    }
                ]
                logging.error(traceback.format_exc())
                continue
            csv_rows += download_info
            error_rows += error_info
            if (
                self.auth is not None
                and self.auth.token is not None
                and (
                    (datetime.datetime.now() - self.auth.auth_time)
                    > datetime.timedelta(minutes=self.auth.token_refresh_minutes)
                )
            ):
                logging.info("Refreshing token...")
                self.auth.refresh_token()
        new_mapping_df = pd.concat([mapping_df, pd.DataFrame(csv_rows)])
        error_df = pd.DataFrame(error_rows)
        return new_mapping_df, error_df

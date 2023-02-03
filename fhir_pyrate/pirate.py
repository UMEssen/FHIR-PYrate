import datetime
import hashlib
import logging
import math
import multiprocessing
import re
import traceback
import warnings
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import fhirpathpy
import pandas as pd
import requests
from dateutil.parser import parse
from requests.adapters import HTTPAdapter, Retry
from requests_cache import CachedSession
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from fhir_pyrate import Ahoy
from fhir_pyrate.util import FHIRObj, string_from_column
from fhir_pyrate.util.bundle_processing_templates import flatten_data, parse_fhir_path

logger = logging.getLogger(__name__)


def create_key(request: requests.PreparedRequest, **kwargs: Any) -> str:
    """
    Creates a unique key for each request URL.

    :param request: The request to create a key for
    :param kwargs: Unused, needed for compatibility with the library
    :return: A string which is a hash of the request
    """
    assert isinstance(request.url, str)
    return hashlib.sha256(request.url.encode()).hexdigest()


class Pirate:
    """
    Main class to query resources using the FHIR API.

    :param base_url: The main URL where the FHIR server is located
    :param auth:  Either an authenticated instance of the Ahoy class, or an authenticated
    requests.Session that can be used to communicate and to query resources
    :param num_processes: The number of processes that should be used to run the query for the
    functions that use multiprocessing
    :param print_request_url: Whether the request URLs should be printed whenever we do a request
    :param time_format: The time format used by the FHIR API
    :param default_count: The default count of results per page used by the server
    :param cache_folder: Whether the requests should be stored for later use, and where
    :param cache_expiry_time: In case the cache is used, when should it expire
    :param retry_requests: You can specify a requests.adapter.Retry instance to retry the requests
    that are failing, an example could be
    retry_requests=Retry(
        total=3, # Retries for a total of three times
        backoff_factor=0.5, # A backoff factor to apply between attempts, such that the requests
        # are not run directly one after the other
        status_forcelist=[500, 502, 503, 504], # HTTP status codes that we should force a retry on
        allowed_methods=["GET"] # Set of uppercased HTTP method verbs that we should retry on
    )
    Complete set of parameters:
    https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html
    :param disable_multiprocessing_requests: Whether to disable multiprocessing for running the
    requests with the FHIR server
    :param disable_multiprocessing_build: Whether to disable multiprocessing when building the
    DataFrame
    :param silence_fhirpath_warning: Whether the FHIR path warning regarding already existing
    expressions should be silenced
    :param optional_get_params: Optional parameters that will be passed to the session's get calls
    """

    FHIRPATH_INVALID_TOKENS = [
        "div",
        "mod",
        "in",
        "and",
        "or",
        "xor",
        "implies",
    ]

    def __init__(
        self,
        base_url: str,
        auth: Optional[Union[requests.Session, Ahoy]],
        num_processes: int = 1,
        print_request_url: bool = False,
        time_format: str = "%Y-%m-%dT%H:%M",
        default_count: int = None,
        cache_folder: Union[str, Path] = None,
        cache_expiry_time: Union[datetime.datetime, int] = -1,  # -1 = does not expire
        retry_requests: Retry = None,
        disable_multiprocessing_requests: bool = False,
        disable_multiprocessing_build: bool = False,
        silence_fhirpath_warning: bool = False,
        optional_get_params: Dict[Any, Any] = None,
    ):
        # Remove the last character if they added it
        url_search = re.search(
            pattern=r"(https?:\/\/([^\/]+))([\w\.\-~\/]*)", string=base_url
        )
        if url_search is None:
            raise ValueError(
                "The given URL does not follow the validation RegEx. Was your URL "
                "written correctly? If it is, please create an issue."
            )
        self.base_url = url_search.group(1)
        self.domain = url_search.group(2)
        self.fhir_app_location = (
            url_search.group(3)
            if len(url_search.group(3)) > 0 and url_search.group(3)[-1] == "/"
            else url_search.group(3) + "/"
        )
        self._close_session_on_exit = False
        if isinstance(auth, Ahoy):
            self.session = auth.session
        elif isinstance(auth, requests.Session):
            self.session = auth
        else:
            self._close_session_on_exit = True
            self.session = requests.session()
        self.disable_multiprocessing_requests = disable_multiprocessing_requests
        self.disable_multiprocessing_build = disable_multiprocessing_build
        self.caching = False
        if cache_folder is not None:
            # TODO: Change this to work with context managers
            session = CachedSession(
                str(Path(cache_folder) / "fhir_pyrate"),
                # cache_control=False,
                # Use Cache-Control response headers for expiration, if available
                expire_after=cache_expiry_time,  # Otherwise expire responses after one day
                allowable_codes=[
                    200,
                    400,
                ],  # Cache 400 responses as a solemn reminder of your failures
                allowable_methods=["GET"],  # Cache whatever HTTP methods you want
                ignored_parameters=["api_key"],
                # # Don't match this request param, and redact if from the cache
                match_headers=[
                    "Accept-Language"
                ],  # Cache a different response per language
                stale_if_error=True,  # In case of request errors, use stale cache data if possible
                key_fn=create_key,
            )
            session.auth = self.session.auth
            self.session = session
            self.caching = True
            self.disable_multiprocessing_requests = True
            logger.warning(
                "Request caching and multiprocessing cannot be run together."
            )
        if self.disable_multiprocessing_requests and self.disable_multiprocessing_build:
            self.num_processes = 1
        else:
            self.num_processes = num_processes

        if retry_requests is not None:
            self.session.mount(self.base_url, HTTPAdapter(max_retries=retry_requests))

        self.optional_get_params = (
            optional_get_params if optional_get_params is not None else {}
        )
        self._print_request_url = print_request_url
        self._time_format = time_format
        self._default_count = default_count
        self.silence_fhirpath_warning = silence_fhirpath_warning

    ##############################
    #      MAIN FUNCTIONS        #
    ##############################

    def get_bundle_total(
        self,
        resource_type: str,
        request_params: Dict[str, Any] = None,
        count_entries: bool = False,
    ) -> Optional[int]:
        """
        Perform a request to return the total amount of bundles for a query.

        :param resource_type: The resource to query, e.g. Patient, DiagnosticReport
        :param request_params: The parameters for the current request
        :param count_entries: Whether the number of entries should be counted instead
        :return: The number of entries for this bundle, either given by the total attribute or by
        counting the number of entries
        """
        return self._get_total_from_bundle(
            bundle=self._get_response(
                self._build_request_url(
                    resource_type=resource_type, request_params=request_params or {}
                )
            ),
            count_entries=count_entries,
        )

    def steal_bundles(
        self,
        resource_type: str,
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
    ) -> Generator[FHIRObj, None, int]:
        """
        Executes a request, iterates through the result pages and returns all the bundles as a
        generator.

        :param resource_type: The resource to be queried, e.g. DiagnosticReport
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :return: A Generator of FHIR bundles containing the queried information
        """
        request_params = {} if request_params is None else request_params.copy()
        with logging_redirect_tqdm():
            return self._get_bundles(
                resource_type=resource_type,
                request_params=request_params,
                num_pages=num_pages,
                silence_tqdm=False,
                tqdm_df_build=False,
            )

    def steal_bundles_to_dataframe(
        self,
        resource_type: str,
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
        build_df_after_query: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Executes a request, iterates through the result pages, and builds a DataFrame with their
        information. The DataFrames are either built after each
        bundle is retrieved, or after we collected all bundles.

        :param resource_type: The resource to be queried, e.g. DiagnosticReport
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param fhir_paths: A list of FHIR paths (https://hl7.org/fhirpath/) to be used to build the
        DataFrame, alternatively, a list of tuples can be used to specify the column name of the
        future column with (column_name, fhir_path). Please refer to the `bundles_to_dataframe`
        functions for notes on how to use the FHIR paths
        :param build_df_after_query: Whether the DataFrame should be built after all bundles have
        been collected, or whether the bundles should be transformed just after retrieving
        :return: A DataFrame per queried resource. In case only once resource is queried, then only
        one dictionary is given back, otherwise a dictionary of (resourceType, DataFrame) is
        returned.
        """
        return self._query_to_dataframe(self._get_bundles)(
            resource_type=resource_type,
            request_params=request_params,
            num_pages=num_pages,
            silence_tqdm=False,
            process_function=process_function,
            fhir_paths=fhir_paths,
            build_df_after_query=build_df_after_query,
            disable_multiprocessing_build=True,
            always_return_dict=False,
        )

    def sail_through_search_space(
        self,
        resource_type: str,
        time_attribute_name: str,
        date_init: Union[str, datetime.date],
        date_end: Union[str, datetime.date],
        request_params: Dict[str, Any] = None,
    ) -> Generator[FHIRObj, None, int]:
        """
        Uses the multiprocessing module to speed up some queries. The time frame is
        divided into multiple time spans (as many as there are processes) and each smaller
        time frame is investigated simultaneously.

        :param resource_type: The resource to query, e.g. Patient, DiagnosticReport
        :param time_attribute_name: The time attribute that should be used to define the
        timespan; e.g. `started` for ImagingStudy, `date` for DiagnosticReport; `_lastUpdated`
        should be able to be used by all queries
        :param date_init: The start of the timespan for `time_attribute_name` (inclusive)
        :param date_end: The end of the timespan for `time_attribute_name` (exclusive)
        :param request_params:  The parameters for the query, e.g. `_count`, `_id`, `_sort`
        :return: A Generator containing FHIR bundles with the queried information for all timespans
        """
        return self._sail_through_search_space(
            resource_type=resource_type,
            time_attribute_name=time_attribute_name,
            date_init=date_init,
            date_end=date_end,
            request_params=request_params,
            tqdm_df_build=False,
        )

    def sail_through_search_space_to_dataframe(
        self,
        resource_type: str,
        time_attribute_name: str,
        date_init: Union[str, datetime.date],
        date_end: Union[str, datetime.date],
        request_params: Dict[str, Any] = None,
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
        build_df_after_query: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Uses the multiprocessing module to speed up some queries. The time frame is
        divided into multiple time spans (as many as there are processes) and each smaller
        time frame is investigated simultaneously. Finally, it builds a DataFrame with the
        information from all timespans. The DataFrames are either built after each
        bundle is retrieved, or after we collected all bundles.

        :param resource_type: The resource to query, e.g. Patient, DiagnosticReport
        :param time_attribute_name: The time attribute that should be used to define the
        timespan; e.g. `started` for ImagingStudy, `date` for DiagnosticReport; `_lastUpdated`
        should be able to be used by all queries
        :param date_init: The start of the timespan for `time_attribute_name` (inclusive)
        :param date_end: The end of the timespan for `time_attribute_name` (exclusive)
        :param request_params:  The parameters for the query, e.g. `_count`, `_id`, `_sort`
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param fhir_paths: A list of FHIR paths (https://hl7.org/fhirpath/) to be used to build the
        DataFrame, alternatively, a list of tuples can be used to specify the column name of the
        future column with (column_name, fhir_path). Please refer to the `bundles_to_dataframe`
        functions for notes on how to use the FHIR paths
        :param build_df_after_query: Whether the DataFrame should be built after all bundles have
        been collected, or whether the bundles should be transformed just after retrieving
        :return: A DataFrame per queried resource for all timestamps. In case only once resource
        is queried, then only one dictionary is given back, otherwise a dictionary of
        (resourceType, DataFrame) is returned.
        """
        return self._query_to_dataframe(self._sail_through_search_space)(
            resource_type=resource_type,
            request_params=request_params,
            time_attribute_name=time_attribute_name,
            date_init=date_init,
            date_end=date_end,
            process_function=process_function,
            fhir_paths=fhir_paths,
            build_df_after_query=build_df_after_query,
            always_return_dict=False,
        )

    def trade_rows_for_bundles(
        self,
        df: pd.DataFrame,
        resource_type: str,
        df_constraints: Dict[
            str, Union[Union[str, Tuple[str, str]], List[Union[str, Tuple[str, str]]]]
        ],
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
    ) -> Generator[FHIRObj, None, int]:
        """
        Go through the rows of a DataFrame (with multiprocessing), run a query and retrieve
        bundles for each row.

        :param df: The DataFrame with the queries
        :param resource_type: The resource to query, e.g. Patient, DiagnosticReport
        :param df_constraints: A dictionary containing a mapping between the FHIR attributes and
        the columns of the input DataFrame, e.g. {"subject" : "fhir_patient_id"}, where subject
        is the FHIR attribute and fhir_patient_id is the name of the column. It is also possible
        to add the system by using a tuple instead of a string, e.g. "code": (
        "http://loinc.org", "loinc_code")
        Possible structures:
        {"code": "code_column"}
        {"code": ("code_system", "code_column")}
        {"date": ["init_date_column", "end_date_column"]}
        {"date": [("ge", "init_date_column"), ("le", "end_date_column")]}
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :return: A Generator containing FHIR bundles with the queried information for all rows
        """
        return self._trade_rows_for_bundles(
            df=df,
            resource_type=resource_type,
            df_constraints=df_constraints,
            request_params=request_params,
            num_pages=num_pages,
            tqdm_df_build=False,
        )

    def trade_rows_for_dataframe(
        self,
        df: pd.DataFrame,
        resource_type: str,
        df_constraints: Dict[
            str, Union[Union[str, Tuple[str, str]], List[Union[str, Tuple[str, str]]]]
        ],
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
        with_ref: bool = True,
        with_columns: List[Union[str, Tuple[str, str]]] = None,
        build_df_after_query: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Go through the rows of a DataFrame (with multiprocessing), run a query, retrieve
        bundles for each row and transform them into a DataFrame.

        The DataFrames can be computed in different ways:
        1. The bundles are retrieved and the DataFrame is computed straight away, which can be
        obtained by setting `build_df_after_query` to False. If `with_ref` is True, then the
        DataFrame is always computed like this.
        2. If `build_df_after_query` is True and `with_ref` is False, then first all bundles
        will be retrieved, and then they will be processed into a DataFrame.

        :param df: The DataFrame with the queries
        :param resource_type: The resource to query, e.g. Patient, DiagnosticReport
        :param df_constraints: A dictionary containing a mapping between the FHIR attributes and
        the columns of the input DataFrame, e.g. {"subject" : "fhir_patient_id"}, where subject
        is the FHIR attribute and fhir_patient_id is the name of the column. It is also possible
        to add the system by using a tuple instead of a string, e.g. "code": (
        "http://loinc.org", "loinc_code")
        Possible structures:
        {"code": "code_column"}
        {"code": ("code_system", "code_column")}
        {"date": ["init_date_column", "end_date_column"]}
        {"date": [("ge", "init_date_column"), ("le", "end_date_column")]}
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned per request,
        the default is -1 (all bundles), with any other value exactly that value of bundles will
        be returned, assuming that there are that many
        :param with_ref: Whether the input columns of `df_constraints` should be added to the
        output DataFrame
        :param with_columns: Whether additional columns from the source DataFrame should be
        added to output DataFrame. The columns from the source DataFrame can be either specified
        as a list of columns `[col1, col2, ...]` or as a list of tuples
        `[(new_name_for_col1, col1), (new_name_for_col2, col2), ...]`
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param fhir_paths: A list of FHIR paths (https://hl7.org/fhirpath/) to be used to build the
        DataFrame, alternatively, a list of tuples can be used to specify the column name of the
        future column with (column_name, fhir_path). Please refer to the `bundles_to_dataframe`
        functions for notes on how to use the FHIR paths
        :param build_df_after_query: Whether the DataFrame should be built after all bundles have
        been collected, or whether the bundles should be transformed just after retrieving
        :return: A DataFrame per queried resource which contains information about all rows.
        In case only once resource is queried, then only one dictionary is given back, otherwise
        a dictionary of (resourceType, DataFrame) is returned.
        """
        with logging_redirect_tqdm():
            if fhir_paths is not None:
                logger.info(
                    f"The selected process_function {process_function.__name__} will be "
                    f"overwritten."
                )
                process_function = self._set_up_fhirpath_function(fhir_paths)
            if not with_ref and not with_columns:
                return self._bundles_to_dataframe(
                    bundles=self._trade_rows_for_bundles(
                        df=df,
                        resource_type=resource_type,
                        df_constraints=df_constraints,
                        request_params=request_params,
                        num_pages=num_pages,
                        tqdm_df_build=not build_df_after_query,
                    ),
                    process_function=process_function,
                    build_df_after_query=build_df_after_query,
                    disable_multiprocessing=self.disable_multiprocessing_build,
                    always_return_dict=False,
                )
            if self.caching and self.num_processes > 1:
                logger.info(
                    "In trade_rows_for_dataframe with multiprocessing, each row is handled by a "
                    "separate process, which sends the request and builds a Dataframe. "
                    "Since caching does not support multiprocessing, this function will now run on "
                    "a single process."
                )
            else:
                logger.info(
                    f"Querying each row of the DataFrame with {self.num_processes} processes."
                )
            request_params = {} if request_params is None else request_params.copy()
            adjusted_constraints = self._adjust_df_constraints(df_constraints)
            req_params_per_sample = self._get_request_params_for_sample(
                df=df,
                request_params=request_params,
                df_constraints=adjusted_constraints,
            )
            # Reformat the with_columns attribute such that it becomes a list of tuples
            # (new_name, current_name)
            # The columns to be added are either as elements in the list
            # [col1, col2, ...]
            # or as second argument of a tuple
            # [(renamed_col_1, col1), ...]
            with_columns_adjusted = (
                [(col, col) if isinstance(col, str) else col for col in with_columns]
                if with_columns is not None
                else []
            )
            # Create a dictionary to rename the columns
            with_columns_rename = {
                col: col_rename for col_rename, col in with_columns_adjusted
            }
            # Also go through the df_constraints, in case they are not in the list for renaming
            for _, list_of_constraints in adjusted_constraints.items():
                for _, value in list_of_constraints:
                    if value not in with_columns_rename:
                        with_columns_rename[value] = value
            # Prepare the inputs that will end up in the final dataframe
            input_params_per_sample = [
                {
                    **{
                        # The name of the parameter will be the same as the column name
                        # The value will be the same as the value in that column for that row
                        value: row[df.columns.get_loc(value)]
                        # Concatenate the given system identifier string with the desired
                        # identifier
                        for _, list_of_constraints in adjusted_constraints.items()
                        for _, value in list_of_constraints
                    },
                    # Add other columns from with_columns
                    **{
                        col: row[df.columns.get_loc(col)]
                        for _, col in with_columns_adjusted
                    },
                }
                for row in df.itertuples(index=False)
            ]
            # Add all the parameters needed by the steal_bundles function
            params_per_sample = [
                {
                    "resource_type": resource_type,
                    "request_params": req_sample,
                    "num_pages": num_pages,
                    "silence_tqdm": True,
                }
                for req_sample in req_params_per_sample
            ]
            final_dfs: Dict[str, List[pd.DataFrame]] = {}
            tqdm_text = f"Query & Build DF ({resource_type})"
            if (
                self.disable_multiprocessing_requests
                or self.disable_multiprocessing_build
            ):
                # If we don't want multiprocessing
                for param, input_param in tqdm(
                    zip(params_per_sample, input_params_per_sample),
                    total=len(params_per_sample),
                    desc=tqdm_text,
                ):
                    # Get the dataframe
                    found_dfs = self._query_to_dataframe(self._get_bundles)(
                        process_function=process_function,
                        build_df_after_query=False,
                        disable_multiprocessing_build=True,
                        always_return_dict=True,
                        **param,
                    )
                    for resource_type, found_df in found_dfs.items():
                        final_dfs.setdefault(resource_type, [])
                        self._copy_existing_columns(
                            df=found_df,
                            input_params=input_param,
                            key_mapping=with_columns_rename,
                        )
                        final_dfs[resource_type].append(found_df)
            else:
                pool = multiprocessing.Pool(self.num_processes)
                results = []
                for param, input_param in tqdm(
                    zip(params_per_sample, input_params_per_sample),
                    total=len(params_per_sample),
                    desc=tqdm_text,
                ):
                    # Add the functions that we want to run
                    results.append(
                        (
                            pool.apply_async(
                                self._bundles_to_dataframe,
                                kwds=dict(
                                    bundles=[b for b in self._get_bundles(**param)],
                                    process_function=process_function,
                                    build_df_after_query=False,
                                    disable_multiprocessing=True,
                                    always_return_dict=True,
                                ),
                            ),
                            input_param,
                        )
                    )
                for async_result, input_param in results:
                    # Get the results and build the dataframes
                    found_dfs = async_result.get()
                    for resource_type, found_df in found_dfs.items():
                        final_dfs.setdefault(resource_type, [])
                        self._copy_existing_columns(
                            df=found_df,
                            input_params=input_param,
                            key_mapping=with_columns_rename,
                        )
                        final_dfs[resource_type].append(found_df)
                pool.close()
                pool.join()
            dfs = {
                resource_type: pd.concat(final_dfs[resource_type], ignore_index=True)
                for resource_type in final_dfs
            }
            return list(dfs.values())[0] if len(dfs) == 1 else dfs

    def bundles_to_dataframe(
        self,
        bundles: Union[List[FHIRObj], Generator[FHIRObj, None, int]],
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Convert a bundle into a DataFrame using either the `flatten_data` function (default),
        FHIR paths or a custom processing function. For the case of `flatten_data` and the FHIR
        paths, each row of the DataFrame will represent a resource. In the custom processing
        function each bundle can be handled as one pleases.

        :param bundles: The bundles to transform
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param fhir_paths: A list of FHIR paths (https://hl7.org/fhirpath/) to be used to build the
        DataFrame, alternatively, a list of tuples can be used to specify the column name of the
        future column with (column_name, fhir_path).
        :return: A DataFrame per queried resource. In case only once resource is queried, then only
        one dictionary is given back, otherwise a dictionary of (resourceType, DataFrame) is
        returned.
        **NOTE 1 on FHIR paths**: The standard also allows some primitive math operations such as
        modulus (`mod`) or integer division (`div`), and this may be problematic if there are
        fields of the resource that use these terms as attributes.
        It is actually the case in many generated
        [public FHIR resources](https://hapi.fhir.org/baseDstu2/DiagnosticReport/133015).
        In this case the term `text.div` cannot be used, and you should use a processing function
        instead (as in 2.).
        **NOTE 2 on FHIR paths**: Since it is possible to specify the column name with a tuple
        `(key, fhir_path)`, it is important to know that if a key is used multiple times for different
        pieces of information but for the same resource, the field will be only filled with the first
        occurence that is not None.
        ```python
        df = search.steal_bundles_to_dataframe(
            resource_type="DiagnosticReport",
            request_params={
                "_count": 1,
                "_include": "DiagnosticReport:subject",
            },
            # CORRECT EXAMPLE
            # In this case subject.reference is None for patient, so all patients will have their Patient.id
            fhir_paths=[("patient", "subject.reference"), ("patient", "Patient.id")],
            # And Patient.id is None for DiagnosticReport, so they will have their subject.reference
            # fhir_paths=[("patient", "Patient.id"), ("patient", "subject.reference")],
            # WRONG EXAMPLE
            # In this case, only the first code will be stored
            # fhir_paths=[("code", "code.coding[0].code"), ("code", "code.coding[1].code")],
            # CORRECT EXAMPLE
            # Whenever we are working with codes, it is usually better to use the `where` argument
            # and to store the values using a meaningful name
            # fhir_paths=[
            #     ("code_abc", "code.coding.where(system = 'ABC').code"),
            #     ("code_def", "code.coding.where(system = 'DEF').code")
            # ],
            num_pages=1,
        )
        """
        with logging_redirect_tqdm():
            if fhir_paths is not None:
                logger.info(
                    f"The selected process_function {process_function.__name__} will be "
                    f"overwritten."
                )
                process_function = self._set_up_fhirpath_function(fhir_paths)
            return self._bundles_to_dataframe(
                bundles=bundles,
                process_function=process_function,
                build_df_after_query=True,
                disable_multiprocessing=self.disable_multiprocessing_build,
                always_return_dict=False,
            )

    @staticmethod
    def smash_rows(
        df: pd.DataFrame,
        group_by_col: str,
        separator: str = ", ",
        unique: bool = False,
        sort: bool = False,
        sort_reverse: bool = False,
    ) -> pd.DataFrame:
        """
        Group a DataFrame by a certain row and summarize all results of the involved rows into a
        single cell, separated by a predefined separator.

        :param df: The DataFrame to smash
        :param group_by_col: The column we should use to group by
        :param separator: The separator for the values
        :param unique: Whether only unique values should be stored
        :param sort: Whether the values should be sorted
        :param sort_reverse: Whether the values should sorted in reverse order
        :return: A DataFrame containing where the rows that have been grouped are now in a single
        row
        """
        return df.groupby(group_by_col, as_index=False).agg(
            {
                col: partial(
                    string_from_column,
                    separator=separator,
                    unique=unique,
                    sort=sort,
                    sort_reverse=sort_reverse,
                )
                for col in df.columns
            }
        )

    ##############################
    #     CONTEXT HANDLING       #
    ##############################

    def __enter__(self) -> "Pirate":
        return self

    def close(self) -> None:
        # Only close the session if it does not come from an authentication class
        if self._close_session_on_exit:
            self.session.close()

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        self.close()

    ##############################
    # REQUEST PARAMETER HANDLING #
    ##############################

    @staticmethod
    def _concat_request_params(request_params: Dict[str, Any]) -> str:
        """
        Concatenates the parameters to create a request string.

        :param request_params: The parameters that should be used for the request
        :return: The concatenated string for the request
        """
        if "history" in request_params or "_id" in request_params:
            if "history" in request_params:
                param = "history"
            else:
                param = "_id"
            found_param = (
                request_params[param]
                if not isinstance(request_params[param], List)
                else next(iter(request_params[param]))
            )
            assert isinstance(found_param, str)
            if "history" in request_params:
                return f"{found_param}/_history"
            else:
                return found_param
        params = [
            f"{k}={v}"
            for k, v in sorted(request_params.items())
            if not isinstance(v, (list, tuple))
        ]
        params += [
            f"{k}={element}"
            for k, v in sorted(request_params.items())
            if isinstance(v, (list, tuple))
            for element in v
        ]
        return "&".join(params)

    @staticmethod
    def _adjust_df_constraints(
        df_constraints: Dict[
            str, Union[Union[str, Tuple[str, str]], List[Union[str, Tuple[str, str]]]]
        ]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Adjust the constraint dictionary to always have the same structure, which makes it easier
        to parse it for other function.

        Possible structures:
        {"code": "code_column"}
        {"code": ("code_system", "code_column")}
        {"date": ["init_date_column", "end_date_column"]}
        {"date": [("ge", "init_date_column"), ("le", "end_date_column")]}

        :param df_constraints: A dictionary that specifies the constraints that should be applied
        during a search query and that refer to a DataFrame
        :return: A standardized request dictionary
        """
        # First make sure that everything is transformed into a dictionary of lists
        df_constraints_list: Dict[str, List[Union[str, Tuple[str, str]]]] = {
            fhir_identifier: (
                [possible_list]
                if not isinstance(possible_list, List)
                else possible_list
            )
            for fhir_identifier, possible_list in df_constraints.items()
        }
        # Then handle the internal tuples
        return {
            fhir_identifier: [
                ("", column_constraint)
                if isinstance(column_constraint, str)
                else (
                    column_constraint[0]
                    + (
                        "%7C"
                        if "http" in column_constraint[0]
                        and "%7C" not in column_constraint[0]
                        else ""
                    ),
                    column_constraint[1],
                )
                for column_constraint in list_of_constraints
            ]
            for fhir_identifier, list_of_constraints in df_constraints_list.items()
        }

    @staticmethod
    def _get_request_params_for_sample(
        df: pd.DataFrame,
        request_params: Dict[str, Any],
        df_constraints: Dict[str, List[Tuple[str, str]]],
    ) -> List[Dict[str, List[str]]]:
        """
        Builds the request parameters for each sample by checking the constraint set on each row.
        The resulting request parameters are given by the general `request_params` and by the
        constraint given by each row. E.g. if df_constraints = {"subject": "patient_id"}, then
        the resulting list will contain {"subject": row.patient_id} for each row of the DataFrame.

        :param df: The DataFrame that contains the constraints that should be applied
        :param request_params: The parameters for the query that do not depend on the DataFrame
        :param df_constraints: A dictionary that specifies the constraints that should be applied
        during a search query and that refer to a DataFrame
        :return: A list of dictionary constraint for each row of the DataFrame
        """
        for _, list_of_constraints in df_constraints.items():
            for _, value in list_of_constraints:
                if df[value].isnull().any():
                    raise ValueError(
                        f"The column {value} contains NaN values, "
                        f"and thus it cannot be used to build queries."
                    )
        return [
            dict(
                {
                    fhir_identifier: [
                        (modifier + str(row[df.columns.get_loc(value)]).split("/")[-1])
                        if fhir_identifier == "_id"
                        else modifier + str(row[df.columns.get_loc(value)])
                        for modifier, value in list_of_constraints
                    ]
                    # Concatenate the given system identifier string with the desired identifier
                    for fhir_identifier, list_of_constraints in df_constraints.items()
                },
                **request_params,
            )
            for row in df.itertuples(index=False)
        ]

    def _get_timespan_list(
        self, date_init: str, date_end: str
    ) -> List[Tuple[str, str]]:
        """
        Divides a timespan into equal parts according to the number of processes selected.

        :param date_init: Beginning of the timespan
        :param date_end: End of the timespan
        :return: A list of sub periods that divide the timespan into equal parts
        """
        timespans = (
            pd.date_range(date_init, date_end, periods=(self.num_processes + 1))
            .strftime(self._time_format)
            .tolist()
        )
        # Convert the list into tuples
        return [(timespans[i], timespans[i + 1]) for i in range(len(timespans) - 1)]

    def _return_count_from_request(
        self, request_params: Dict[str, Any]
    ) -> Optional[int]:
        """
        Return the number of expected resources per page. If count has been defined in the
        request parameters, return it, otherwise choose the default count that has been given as
        input to the class.

        :param request_params: The parameters for the current request
        :return: The number of resources that shall be returned for every page
        """
        return (
            int(request_params["_count"])
            if "_count" in request_params
            else self._default_count
        )

    ##############################
    #       BUNDLE HANDLING      #
    ##############################

    def _get_response(self, request_url: str) -> Optional[FHIRObj]:
        """
        Performs the API request and returns the response as a dictionary.

        :param request_url: The request string
        :return: A FHIR bundle
        """
        try:
            response = self.session.get(request_url, **self.optional_get_params)
            if self._print_request_url:
                tqdm.write(request_url)
            response.raise_for_status()
            json_response = response.json()
            # If it's a bundle return it
            if json_response.get("resourceType") == "Bundle":
                return FHIRObj(**json_response)
            # Otherwise it's a read operation (are there other options?)
            # and we should convert it to a bundle for the sake of consistency
            else:
                return FHIRObj(
                    **{
                        "resourceType": "Bundle",
                        "type": "read",
                        "total": 1,
                        "entry": [
                            {
                                "full_url": request_url,
                                "resource": json_response,
                            }
                        ],
                    }
                )
        except Exception:
            # Leave this to be able to quickly see the errors
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def _get_total_from_bundle(
        bundle: Optional[FHIRObj],
        count_entries: bool = False,
    ) -> Optional[int]:
        """
        Return the total attribute of a bundle or the number of entries.

        :param bundle: The bundle for which we need to find the total
        :param count_entries: Whether the number of entries should be counted instead
        :return: The total attribute for this bundle or the total number of entries
        """
        if bundle is not None:
            if count_entries and bundle.entry is not None:
                return len(bundle.entry)
            if bundle.total is not None:
                assert isinstance(bundle.total, int)
                return bundle.total
        return None

    def _build_request_url(
        self, resource_type: str, request_params: Dict[str, Any]
    ) -> str:
        """
        Use the resource type and the request parameters to build the final request URL.

        :param resource_type: The resource to call
        :param request_params: The parameters for the request
        :return: The URL for the request
        """
        request_params_string = self._concat_request_params(request_params)
        return (
            f"{self.base_url}{self.fhir_app_location}{resource_type}"
            f"{'/' if ('history' in request_params or '_id' in request_params) else '?'}{request_params_string}"
        )

    def _get_bundles(
        self,
        resource_type: str,
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
        silence_tqdm: bool = False,
        tqdm_df_build: bool = False,
    ) -> Generator[FHIRObj, None, int]:
        """
        Executes a request, iterates through the result pages and returns all the bundles as a
        generator.
        Additionally, some checks are performed, and the corresponding warnings are returned:
        - Whether a sorting has been defined
        - Whether the current bundle is empty

        :param resource_type: The resource to be queried, e.g. DiagnosticReport
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :param silence_tqdm: Whether tqdm should be disabled
        :param tqdm_df_build: Whether this function is being used by a wrapper to also build a
        DataFrame, in such case the "Build DF" string will be added to the tdqm text
        :return: A Generator of FHIR bundles containing the queried information
        """
        if num_pages == 0:
            return 0
        current_params = {} if request_params is None else request_params
        bundle = self._get_response(
            self._build_request_url(resource_type, current_params)
        )
        bundle_total: Union[int, float] = num_pages
        total = self._get_total_from_bundle(bundle, count_entries=False)
        if bundle_total == -1:
            n_entries = self._get_total_from_bundle(bundle, count_entries=True)
            if total and n_entries:
                bundle_total = math.ceil(total / n_entries)
            else:
                bundle_total = math.inf
        if (
            "history" not in (request_params or {})
            and "_id" not in (request_params or {})
            and bundle_total != math.inf
            and bundle_total > 1
            and not any(k == "_sort" for k, _ in current_params.items())
        ):
            warnings.warn(
                f"We detected multiple pages (approximately {bundle_total} pages) but "
                f"no sorting method has been defined, which may yield incorrect results. "
                f"We will set the sorting parameter to _id."
            )
            current_params["_sort"] = "_id"
            request_params_string = self._concat_request_params(current_params)
            bundle = self._get_response(
                f"{self.base_url}{self.fhir_app_location}{resource_type}?{request_params_string}"
            )

        progress_bar = tqdm(
            disable=silence_tqdm,
            desc=f"Query & Build DF ({resource_type})"
            if tqdm_df_build
            else f"Query ({resource_type})",
            total=bundle_total,
        )
        bundle_iter = 0
        while bundle is not None:
            progress_bar.update()
            # Return our bundle
            yield bundle
            bundle_iter += 1
            # Find the next page, if it exists
            next_link_url = next(
                (link.url for link in bundle.link or [] if link.relation == "next"),
                None,
            )
            if next_link_url is None or (0 < num_pages <= bundle_iter):
                break
            else:
                # Re-assign bundle and start new iteration
                bundle = self._get_response(
                    f"{self.base_url}{next_link_url}"
                    if self.domain not in next_link_url
                    else next_link_url  # on HAPI the server
                )
        progress_bar.close()
        return bundle_iter

    def _get_bundles_for_timespan(
        self,
        resource_type: str,
        request_params: Dict[str, Any],
        time_attribute_name: str,
        timespan: Tuple[str, str],
        num_pages: int,
        silence_tqdm: bool,
        tqdm_df_build: bool,
    ) -> Generator[FHIRObj, None, int]:
        """
        Wrapper function that sets the `time_attribute_name` date parameters for the
        `sail_through_search_space function`.
        """
        request_params[time_attribute_name] = (
            f"ge{timespan[0]}",
            f"lt{timespan[1]}",
        )
        return self._get_bundles(
            resource_type=resource_type,
            request_params=request_params,
            num_pages=num_pages,
            silence_tqdm=silence_tqdm,
            tqdm_df_build=tqdm_df_build,
        )

    @staticmethod
    def _generator_to_list(f: Callable, *args: Any, **kwargs: Any) -> List[FHIRObj]:
        """
        Wrapper function that converts the result of a function returning a generator to a list.
        """
        return list(f(*args, **kwargs))

    ##############################
    #      QUERY HANDLING        #
    ##############################

    def _run_multiquery(
        self,
        func: Callable,
        query_params: List[Any],
        tqdm_text: str,
    ) -> Generator[FHIRObj, None, int]:
        n_bundles = 0
        if self.disable_multiprocessing_requests:
            for param in tqdm(query_params, total=len(query_params), desc=tqdm_text):
                for bundle in func(param):
                    yield bundle
                    n_bundles += 1
        else:
            pool = multiprocessing.Pool(processes=self.num_processes)
            for bundles_per_query in tqdm(
                # TODO: Can this be done without partial?
                pool.imap(partial(self._generator_to_list, func), query_params),
                total=len(query_params),
                desc=tqdm_text,
            ):
                for bundle in bundles_per_query:
                    yield bundle
                    n_bundles += 1
            pool.close()
            pool.join()
        return n_bundles

    def _sail_through_search_space(
        self,
        resource_type: str,
        time_attribute_name: str,
        date_init: Union[str, datetime.date],
        date_end: Union[str, datetime.date],
        request_params: Dict[str, Any] = None,
        tqdm_df_build: bool = False,
    ) -> Generator[FHIRObj, None, int]:
        """
        Uses the multiprocessing module to speed up some queries. The time frame is
        divided into multiple time spans (as many as there are processes) and each smaller
        time frame is investigated simultaneously.

        :param resource_type: The resource to query, e.g. Patient, DiagnosticReport
        :param time_attribute_name: The time attribute that should be used to define the
        timespan; e.g. started for ImagingStudy, date for DiagnosticReport
        :param date_init: The start of the timespan for time_attribute_name
        :param date_end: The end of the timespan for time_attribute_name
        :param request_params:  The parameters for the query, e.g. _count, _id
        :param tqdm_df_build: Whether this function is being used by a wrapper to also build a
        DataFrame, in such case the "Build DF" string will be added to the tdqm text
        :return: A Generator containing FHIR bundles with the queried information for all timespans
        """
        # Check if the date parameters that we use for multiprocessing are already included in
        # the request parameters
        request_params = {} if request_params is None else request_params.copy()
        search_division_params = [
            k for k in request_params.keys() if k == time_attribute_name
        ]
        # If they are, remove them and issue a warning
        with logging_redirect_tqdm():
            if len(search_division_params) > 0:
                warnings.warn(
                    f"Detected use of parameter {time_attribute_name} "
                    f"in the request parameters. Please use the date_init (inclusive) and "
                    f"date_end (exclusive) parameters instead."
                )
                # Remove all elements that refer to a date
                request_params = {
                    key: request_params[key]
                    for key in request_params
                    if key not in search_division_params
                }
            if not isinstance(date_init, datetime.date):
                date_init = parse(date_init)
            if not isinstance(date_end, datetime.date):
                date_end = parse(date_end)
            date_init = date_init.strftime(self._time_format)
            date_end = date_end.strftime(self._time_format)
            # Copy the dictionary to run a first test
            logger.info(
                f"Running sail_through_search_space with {self.num_processes} processes."
            )
            # Divide the current time period into smaller spans
            timespans = self._get_timespan_list(date_init, date_end)
            func: Callable = partial(
                self._get_bundles_for_timespan,
                resource_type,
                request_params,
                time_attribute_name,
                num_pages=-1,
                silence_tqdm=True,
                tqdm_df_build=tqdm_df_build,
            )
            return self._run_multiquery(
                func=func,
                query_params=timespans,
                tqdm_text=f"Query Timespans & Build DF ({resource_type})"
                if tqdm_df_build
                else f"Query Timespans ({resource_type})",
            )

    def _trade_rows_for_bundles(
        self,
        df: pd.DataFrame,
        resource_type: str,
        df_constraints: Dict[
            str, Union[Union[str, Tuple[str, str]], List[Union[str, Tuple[str, str]]]]
        ],
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
        tqdm_df_build: bool = False,
    ) -> Generator[FHIRObj, None, int]:
        """
        Go through the rows of a DataFrame (with multiprocessing), run a query and retrieve
        bundles for each row.

        :param df: The DataFrame with the queries
        :param resource_type: The resource to query, e.g. Patient, DiagnosticReport
        :param df_constraints: A dictionary containing a mapping between the FHIR attributes and
        the columns of the input DataFrame, e.g. {"subject" : "fhir_patient_id"}, where subject
        is the FHIR attribute and fhir_patient_id is the name of the column. It is also possible
        to add the system by using a tuple instead of a string, e.g. "code": (
        "http://loinc.org", "loinc_code")
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :param tqdm_df_build: Whether this function is being used by a wrapper to also build a
        DataFrame, in such case the "Build DF" string will be added to the tdqm text
        :return: A Generator containing FHIR bundles with the queried information for all rows
        """
        request_params = {} if request_params is None else request_params.copy()
        with logging_redirect_tqdm():
            logger.info(
                f"Querying each row of the DataFrame with {self.num_processes} processes."
            )
            request_params_per_sample = self._get_request_params_for_sample(
                df=df,
                request_params=request_params,
                df_constraints=self._adjust_df_constraints(df_constraints),
            )
            func: Callable = partial(
                self._get_bundles,
                resource_type,
                num_pages=num_pages,
                silence_tqdm=True,
            )
            return self._run_multiquery(
                func=func,
                query_params=request_params_per_sample,
                tqdm_text=f"Query Rows & Build DF ({resource_type})"
                if tqdm_df_build
                else f"Query Rows ({resource_type})",
            )

    @staticmethod
    def _copy_existing_columns(
        df: pd.DataFrame,
        input_params: Dict[str, str],
        key_mapping: Dict[str, str],
    ) -> None:
        """
        Copy the existing columns into the new DataFrame.

        :param df: The DataFrame that was generated from the bundles
        :param input_params: The input columns that should be added to the new DataFrame
        :param key_mapping: A dictionary that can be used to change the name of the input columns
        """
        # Add the key from the input
        for key, value in input_params.items():
            if key_mapping[key] in df.columns:
                warnings.warn(
                    f"A column with name {key_mapping[key]} already exists in the output"
                    f"DataFrame, and the column {key} will not be copied."
                )
            else:
                df[key_mapping[key]] = value

    def _set_up_fhirpath_function(
        self, fhir_paths: List[Union[str, Tuple[str, str]]]
    ) -> Callable:
        """
        Prepares and compiles the FHIRPath and sets them as the processing function for building
        the DataFrames.
        :param fhir_paths: A list of FHIR paths (https://hl7.org/fhirpath/) to be used to build the
        DataFrame, alternatively, a list of tuples can be used to specify the column name of the
        future column with (column_name, fhir_path).
        :return: A function that converts a bundle into a list of information extracted with the
        FHIR paths.
        """
        fhir_paths_with_name = [
            (path[0], path[1]) if isinstance(path, tuple) else (path, path)
            for path in fhir_paths
        ]
        if not self.silence_fhirpath_warning:
            for _, path in fhir_paths_with_name:
                for token in self.FHIRPATH_INVALID_TOKENS:
                    if (
                        re.search(pattern=rf"{token}[\.\[]|[\.\]]{token}$", string=path)
                        is not None
                    ):
                        warnings.warn(
                            f"You are using the term {token} in of your FHIR path {path}. "
                            f"Please keep in mind that this token can be used a function according "
                            f"to the FHIRPath specification (https://hl7.org/fhirpath/), which "
                            f"means that it will not be interpreted as path expression. "
                            f"E.g., if you are using text.div, it will not work, because 'div' is "
                            f"already a string that can be used for integer division. "
                            f"If you really want to do this, please use processing functions "
                            f"instead. If you are using the FHIRPath expressions correctly as "
                            f"they are intended, you can silence the warning when "
                            f"initializing the class."
                        )
        fhir_paths_with_name = [
            (name, fhirpathpy.compile(path=path)) for name, path in fhir_paths_with_name
        ]
        return partial(parse_fhir_path, compiled_fhir_paths=fhir_paths_with_name)

    def _bundles_to_dataframe(
        self,
        bundles: Union[List[FHIRObj], Generator[FHIRObj, None, int]],
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        build_df_after_query: bool = False,
        disable_multiprocessing: bool = False,
        always_return_dict: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Convert a bundle into a DataFrame using either the `flatten_data` function (default),
        FHIR paths or a custom processing function. For the case of `flatten_data` and the FHIR
        paths, each row of the DataFrame will represent a resource. In the custom processing
        function each bundle can be handled as one pleases.

        :param bundles: The bundles to transform
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param build_df_after_query: Whether the DataFrame should be built after all bundles have
        been collected, or whether the bundles should be transformed just after retrieving
        :param disable_multiprocessing: Whether the bundles should be processed sequentially
        :return: A DataFrame per queried resource. In case only once resource is queried, then only
        one dictionary is given back, otherwise a dictionary of (resourceType, DataFrame) is
        returned.
        """
        if disable_multiprocessing:
            processed_bundles = [process_function(bundle) for bundle in bundles]
        else:
            # TODO: It could be that this never makes sense
            pool = multiprocessing.Pool(self.num_processes)
            if build_df_after_query or isinstance(bundles, List):
                bundles = list(bundles)
                processed_bundles = [
                    bundle_output
                    for bundle_output in tqdm(
                        pool.imap(process_function, bundles),
                        total=len(bundles),
                        desc="Build DF",
                    )
                ]
            else:
                processed_bundles = [
                    bundle_output
                    for bundle_output in pool.imap(process_function, bundles)
                ]
            pool.close()
            pool.join()
        results: Dict[str, List[Dict[str, Any]]] = {}
        for bundle_output in processed_bundles:
            if isinstance(bundle_output, List):
                bundle_output = {"SingleResource": bundle_output}
            for resource_type, records in bundle_output.items():
                results.setdefault(resource_type, [])
                results[resource_type] += records
        dfs = {
            resource_type: pd.DataFrame(results[resource_type]).dropna(
                axis=1, how="all"
            )
            for resource_type in results
        }
        if always_return_dict:
            return dfs
        else:
            return list(dfs.values())[0] if len(dfs) == 1 else dfs

    def _query_to_dataframe(
        self,
        bundles_function: Callable,
    ) -> Callable:
        """
        Wrapper function that can be used to transform any function return Lists/Generators of
        bundles into DataFrames.

        :param bundles_function: The function that returns a Generator/List of bundles and that
        can be used to build the DataFrame
        :return: A DataFrame containing the queried information
        """

        def wrap(
            process_function: Callable[[FHIRObj], Any] = flatten_data,
            fhir_paths: List[Union[str, Tuple[str, str]]] = None,
            build_df_after_query: bool = False,
            disable_multiprocessing_build: bool = False,
            always_return_dict: bool = False,
            *args: Any,
            **kwargs: Any,
        ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
            with logging_redirect_tqdm():
                if fhir_paths is not None:
                    logger.info(
                        f"The selected process_function {process_function.__name__} will be "
                        f"overwritten."
                    )
                    process_function = self._set_up_fhirpath_function(fhir_paths)
                return self._bundles_to_dataframe(
                    bundles=bundles_function(
                        *args, **kwargs, tqdm_df_build=not build_df_after_query
                    ),
                    process_function=process_function,
                    build_df_after_query=build_df_after_query,
                    disable_multiprocessing=disable_multiprocessing_build,
                    always_return_dict=always_return_dict,
                )

        return wrap

    def query_to_dataframe(
        self,
        bundles_function: Callable,
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
        build_df_after_query: bool = False,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Wrapper function that given any of the functions that return bundles, builds the
        DataFrame straight away.
        :param bundles_function: The function that should be used to get the bundles,
        e.g. self.sail_through_search_space, trade_rows_for_bundles
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param fhir_paths: A list of FHIR paths (https://hl7.org/fhirpath/) to be used to build the
        DataFrame, alternatively, a list of tuples can be used to specify the column name of the
        future column with (column_name, fhir_path). Please refer to the `bundles_to_dataframe`
        functions for notes on how to use the FHIR paths.
        :param build_df_after_query: Whether the DataFrame should be built after all bundles have
        been collected, or whether the bundles should be transformed just after retrieving
        :param merge_on: Whether to merge the results on a certain row after computing. This is
        useful when using includes, if you store the IDs on the same column you can use that column
        to merge all the rows into one, example below
        :param kwargs: The arguments that will be passed to the `bundles_function` function,
        please refer to the documentation of the respective methods.
        :return: A pandas DataFrame containing the queried information
        The following example will initially return one row for each entry, but using
        `group_row="patient_id"` we choose a column to run the merge on. This will merge the
        columns that contain values that for the others are empty, having then one row representing
        one patient.
        ```
        df = search.query_to_dataframe(
            bundles_function=search.steal_bundles,
            resource_type="Patient",
            request_params={
                "_sort": "_id",
                "_count": 10,
                "birthdate": "ge1990",
                "_revinclude": "Condition:subject",
            },
            fhir_paths=[
                ("patient_id", "Patient.id"),
                ("patient_id", "Condition.subject.reference.replace('Patient/', '')"),
                "Patient.gender",
                "Condition.code.coding.code",
            ],
            num_pages=1,
            group_row="patient_id"
        )
        ```
        """
        if bundles_function == self.steal_bundles:
            return self.steal_bundles_to_dataframe(
                **kwargs,
                process_function=process_function,
                fhir_paths=fhir_paths,
                build_df_after_query=build_df_after_query,
            )
        elif bundles_function == self.sail_through_search_space:
            return self.sail_through_search_space_to_dataframe(
                **kwargs,
                process_function=process_function,
                fhir_paths=fhir_paths,
                build_df_after_query=build_df_after_query,
            )
        elif bundles_function == self.trade_rows_for_bundles:
            return self.trade_rows_for_dataframe(
                **kwargs,
                process_function=process_function,
                fhir_paths=fhir_paths,
                with_ref=False,
                build_df_after_query=build_df_after_query,
            )
        else:
            raise ValueError(
                f"The given function {bundles_function.__name__} "
                f"cannot be used to obtain a dataframe."
            )

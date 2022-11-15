import datetime
import hashlib
import json
import logging
import math
import multiprocessing
import re
import traceback
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import fhirpathpy
import pandas as pd
import requests
from dateutil.parser import parse
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from fhir_pyrate import Ahoy
from fhir_pyrate.util import FHIRObj, string_from_column
from fhir_pyrate.util.bundle_processing_templates import flatten_data, parse_fhir_path

logger = logging.getLogger(__name__)


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
    :param bundle_cache_folder: Whether bundles should be stored for later use, and where
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
        bundle_cache_folder: Union[str, Path] = None,
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
        self.optional_get_params = (
            optional_get_params if optional_get_params is not None else {}
        )
        self.num_processes = num_processes
        self._print_request_url = print_request_url
        self._time_format = time_format
        self._default_count = default_count
        self.bundle_cache_folder = None
        self.silence_fhirpath_warning = silence_fhirpath_warning
        if bundle_cache_folder is not None:
            logger.warning(
                "Bundle caching is a beta feature. This has not yet been extensively "
                "tested and does not have any cache invalidation mechanism."
            )
            self.bundle_cache_folder = Path(bundle_cache_folder)
            self.bundle_cache_folder.mkdir(parents=True, exist_ok=True)

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
        read_from_cache: bool = False,
    ) -> Generator[FHIRObj, None, int]:
        """
        Executes a request, iterates through the result pages and returns all the bundles as a
        generator. If bundle caching is activated and `read_from_cache` is true,
        then the bundles will be read from file instead.

        :param resource_type: The resource to be queried, e.g. DiagnosticReport
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :return: A Generator of FHIR bundles containing the queried information
        """
        request_params = {} if request_params is None else request_params.copy()
        with logging_redirect_tqdm():
            return self._bundle_fn(
                resource_type=resource_type,
                request_params=request_params,
                num_pages=num_pages,
                silence_tqdm=False,
                read_from_cache=read_from_cache,
                tqdm_df_build=False,
            )

    def steal_bundles_to_dataframe(
        self,
        resource_type: str,
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
        read_from_cache: bool = False,
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
        merge_on: str = None,
        build_df_after_query: bool = False,
    ) -> pd.DataFrame:
        """
        Executes a request, iterates through the result pages, and builds a DataFrame with their
        information. If bundle caching is activated and `read_from_cache` is true,
        then the bundles will be read from file instead. The DataFrames are either built after each
        bundle is retrieved, or after we collected all bundles.

        :param resource_type: The resource to be queried, e.g. DiagnosticReport
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param fhir_paths: A list of FHIR paths (https://hl7.org/fhirpath/) to be used to build the
        DataFrame, alternatively, a list of tuples can be used to specify the column name of the
        future column with (column_name, fhir_path). Please refer to the `bundles_to_dataframe`
        functions for notes on how to use the FHIR paths
        :param merge_on: Whether to merge the results on a certain row after computing. This is
        useful when using includes, if you store the IDs on the same column you can use that column
        to merge all the rows into one, an example is given in `bundles_to_dataframe`
        :param build_df_after_query: Whether the DataFrame should be built after all bundles have
        been collected, or whether the bundles should be transformed just after retrieving
        :return: A DataFrame containing the queried information
        """
        return self._query_to_dataframe(self._bundle_fn)(
            resource_type=resource_type,
            request_params=request_params,
            num_pages=num_pages,
            silence_tqdm=False,
            read_from_cache=read_from_cache,
            process_function=process_function,
            fhir_paths=fhir_paths,
            merge_on=merge_on,
            build_df_after_query=build_df_after_query,
            disable_post_multiprocessing=True,
        )

    def sail_through_search_space(
        self,
        resource_type: str,
        time_attribute_name: str,
        date_init: Union[str, datetime.date],
        date_end: Union[str, datetime.date],
        request_params: Dict[str, Any] = None,
        read_from_cache: bool = False,
        disable_multiprocessing: bool = False,
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
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :param disable_multiprocessing: If true, the bundles will be processed sequentially and
        then returned as a Generator; if false, the bundles for each query will be first converted
        to a list (because Generators cannot be pickled, and thus cannot be used by multiprocessing),
        and then they will be yielded as a Generator
        :return: A Generator containing FHIR bundles with the queried information for all timespans
        """
        return self._sail_through_search_space(
            resource_type=resource_type,
            time_attribute_name=time_attribute_name,
            date_init=date_init,
            date_end=date_end,
            request_params=request_params,
            read_from_cache=read_from_cache,
            disable_multiprocessing=disable_multiprocessing,
            tqdm_df_build=False,
        )

    def sail_through_search_space_to_dataframe(
        self,
        resource_type: str,
        time_attribute_name: str,
        date_init: Union[str, datetime.date],
        date_end: Union[str, datetime.date],
        request_params: Dict[str, Any] = None,
        read_from_cache: bool = False,
        disable_multiprocessing: bool = False,
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
        merge_on: str = None,
        build_df_after_query: bool = False,
    ) -> pd.DataFrame:
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
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :param disable_multiprocessing: If true, the bundles will be processed sequentially and
        then returned as a Generator; if false, the bundles for each query will be first converted
        to a list (because Generators cannot be pickled, and thus cannot be used by multiprocessing),
        and then they will be yielded as a Generator
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param fhir_paths: A list of FHIR paths (https://hl7.org/fhirpath/) to be used to build the
        DataFrame, alternatively, a list of tuples can be used to specify the column name of the
        future column with (column_name, fhir_path). Please refer to the `bundles_to_dataframe`
        functions for notes on how to use the FHIR paths
        :param merge_on: Whether to merge the results on a certain row after computing. This is
        useful when using includes, if you store the IDs on the same column you can use that column
        to merge all the rows into one, an example is given in `bundles_to_dataframe`
        :param build_df_after_query: Whether the DataFrame should be built after all bundles have
        been collected, or whether the bundles should be transformed just after retrieving
        :return: A DataFrame containing FHIR bundles with the queried information for all timespans
        """
        return self._query_to_dataframe(self._sail_through_search_space)(
            resource_type=resource_type,
            request_params=request_params,
            time_attribute_name=time_attribute_name,
            date_init=date_init,
            date_end=date_end,
            read_from_cache=read_from_cache,
            disable_multiprocessing=disable_multiprocessing,
            process_function=process_function,
            fhir_paths=fhir_paths,
            merge_on=merge_on,
            build_df_after_query=build_df_after_query,
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
        read_from_cache: bool = False,
        disable_multiprocessing: bool = False,
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
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :param disable_multiprocessing: If true, the bundles will be processed sequentially and
        then returned as a Generator; if false, the bundles for each query will be first converted
        to a list (because Generators cannot be pickled, and thus cannot be used by multiprocessing),
        and then they will be yielded as a Generator
        :return: A Generator containing FHIR bundles with the queried information for all rows
        """
        return self._trade_rows_for_bundles(
            df=df,
            resource_type=resource_type,
            df_constraints=df_constraints,
            request_params=request_params,
            num_pages=num_pages,
            read_from_cache=read_from_cache,
            disable_multiprocessing=disable_multiprocessing,
            tqdm_df_build=False,
        )

    def trade_rows_for_dataframe_with_ref(
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
        merge_on: str = None,
        read_from_cache: bool = False,
        disable_multiprocessing: bool = False,
    ) -> pd.DataFrame:
        logger.warning(
            "The trade_rows_for_dataframe_with_ref function is deprecated, please use "
            "trade_rows_for_dataframe(..., with_ref=True) instead."
        )
        return self.trade_rows_for_dataframe(
            df=df,
            resource_type=resource_type,
            df_constraints=df_constraints,
            process_function=process_function,
            fhir_paths=fhir_paths,
            request_params=request_params,
            num_pages=num_pages,
            with_ref=True,
            merge_on=merge_on,
            read_from_cache=read_from_cache,
            disable_multiprocessing=disable_multiprocessing,
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
        with_ref: bool = False,
        with_columns: List[Union[str, Tuple[str, str]]] = None,
        merge_on: str = None,
        read_from_cache: bool = False,
        disable_multiprocessing: bool = False,
        build_df_after_query: bool = False,
    ) -> pd.DataFrame:
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
        :param merge_on: Whether to merge the results on a certain row after computing. This is
        useful when using includes, if you store the IDs on the same column you can use that column
        to merge all the rows into one, an example is given in `bundles_to_dataframe`
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :param disable_multiprocessing: Whether the rows should be processed sequentially or in
        parallel
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param fhir_paths: A list of FHIR paths (https://hl7.org/fhirpath/) to be used to build the
        DataFrame, alternatively, a list of tuples can be used to specify the column name of the
        future column with (column_name, fhir_path). Please refer to the `bundles_to_dataframe`
        functions for notes on how to use the FHIR paths
        :param build_df_after_query: Whether the DataFrame should be built after all bundles have
        been collected, or whether the bundles should be transformed just after retrieving
        :return: A DataFrame containing FHIR bundles with the queried information for all rows
        and if requested some columns containing the original constraints
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
                        read_from_cache=read_from_cache,
                        disable_multiprocessing=disable_multiprocessing,
                        tqdm_df_build=not build_df_after_query,
                    ),
                    process_function=process_function,
                    merge_on=merge_on,
                    build_df_after_query=build_df_after_query,
                    disable_multiprocessing=disable_multiprocessing,
                )
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
            # The parameters used for post-processing (bundles to dataframe)
            params_for_post = {
                "process_function": process_function,
                "build_df_after_query": False,
                (
                    "disable_post_multiprocessing"
                    if disable_multiprocessing
                    else "disable_multiprocessing"
                ): True,  # The multiprocessing already happens on the rows
            }
            # Add all the parameters needed by the steal_bundles function
            params_per_sample = [
                {
                    "resource_type": resource_type,
                    "request_params": req_sample,
                    "num_pages": num_pages,
                    "silence_tqdm": True,
                    "read_from_cache": read_from_cache,
                }
                for req_sample in req_params_per_sample
            ]
            found_dfs = []
            tqdm_text = f"Query & Build DF ({resource_type})"
            # TODO: Can the merge_on be run for each smaller DataFrame?
            #  is there the possibility to have resources referring to the same thing in
            #  different bundles?
            if disable_multiprocessing:
                # If we don't want multiprocessing
                for param, input_param in tqdm(
                    zip(params_per_sample, input_params_per_sample),
                    total=len(params_per_sample),
                    desc=tqdm_text,
                ):
                    # Get the dataframe
                    found_df = self._query_to_dataframe(self._bundle_fn)(
                        **param, **params_for_post
                    )
                    self._copy_existing_columns(
                        df=found_df,
                        input_params=input_param,
                        key_mapping=with_columns_rename,
                    )
                    found_dfs.append(found_df)
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
                                args=[self._bundle_fn_to_list(**param)],
                                kwds=params_for_post,
                            ),
                            input_param,
                        )
                    )
                for async_result, input_param in results:
                    # Get the results and build the dataframes
                    found_df = async_result.get()
                    self._copy_existing_columns(
                        df=found_df,
                        input_params=input_param,
                        key_mapping=with_columns_rename,
                    )
                    found_dfs.append(found_df)
                pool.close()
                pool.join()
            df = pd.concat(found_dfs, ignore_index=True)
            return (
                df
                if merge_on is None or len(df) == 0
                else self.merge_on_col(df, merge_on)
            )

    def bundles_to_dataframe(
        self,
        bundles: Union[List[FHIRObj], Generator[FHIRObj, None, int]],
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
        merge_on: str = None,
    ) -> pd.DataFrame:
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
        :param merge_on: Whether to merge the results on a certain row after computing. This is
        useful when using includes, if you store the IDs on the same column you can use that column
        to merge all the rows into one, example below
        :return: A pandas DataFrame containing the queried information
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
        ```

        The following example will initially return one row for each entry, but using
        `group_row="patient_id"` we choose a column to run the merge on. This will merge the
        columns that contain values that for the others are empty, having then one row representing
        one patient.
        ```
        df = search.trade_rows_for_dataframe(
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
            merge_on="patient_id"
        )
        ```
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
                merge_on=merge_on,
                build_df_after_query=True,
            )

    @staticmethod
    def merge_on_col(df: pd.DataFrame, merge_on: str) -> pd.DataFrame:
        """
        Merges rows from different resources on a given attribute.
        :param df: The DataFrame where the merge should be applied
        :param merge_on: Whether to merge the results on a certain row after computing. This is
        useful when using includes, if you store the IDs on the same column you can use that column
        to merge all the rows into one
        :return: A DataFrame where the rows having the same `merge_on` attribute are merged.
        """
        # TODO: Could probably be done more efficiently?
        new_df = df[merge_on]
        for col in df.columns:
            if col == merge_on:
                continue
            new_df = pd.merge(
                left=new_df,
                right=df.loc[~df[col].isna(), [merge_on, col]],
                how="outer",
            )
        new_df = new_df.loc[new_df.astype(str).drop_duplicates().index]
        return new_df.reset_index(drop=True)

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
            for k, v in request_params.items()
            if not isinstance(v, (list, tuple))
        ]
        params += [
            f"{k}={element}"
            for k, v in request_params.items()
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
                    + ("%7C" if "http" in column_constraint[0] else ""),
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
            json_response = FHIRObj(**response.json())
            return json_response
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
        read_from_cache: bool = False,
        tqdm_df_build: bool = False,
    ) -> Generator[FHIRObj, None, int]:
        """
        Executes a request, iterates through the result pages and returns all the bundles as a
        generator.
        Additionally, some checks are performed, and the corresponding warnings are returned:
        - Whether the `read_from_cache` is in use
        - Whether a sorting has been defined
        - Whether the current bundle is empty

        :param resource_type: The resource to be queried, e.g. DiagnosticReport
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :param silence_tqdm: Whether tqdm should be disabled
        :param read_from_cache: The parameter is not actually used, it is there to keep
        consistency with the `_cache_bundles` function.
        :param tqdm_df_build: Whether this function is being used by a wrapper to also build a
        DataFrame, in such case the "Build DF" string will be added to the tdqm text
        :return: A Generator of FHIR bundles containing the queried information
        """
        if read_from_cache:
            logger.warning(
                "You are trying to read from cache without having specified a bundle caching "
                "folder, so the caching is not activated and the bundles will not be read"
                "from cache."
            )
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
            logger.warning(
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
            if next_link_url is None or bundle_iter >= bundle_total:
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

    def _cache_bundles(
        self,
        resource_type: str,
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
        silence_tqdm: bool = False,
        read_from_cache: bool = False,
        tqdm_df_build: bool = False,
    ) -> Generator[FHIRObj, None, int]:
        """
        Executes a request, iterates through the result pages and returns all the bundles as a
        generator. If bundle caching is activated and `read_from_cache` is true,
        then the bundles will be read from file instead.

        :param resource_type: The resource to be queried, e.g. DiagnosticReport
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :param silence_tqdm: Whether tqdm should be disabled
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :param tqdm_df_build: Whether this function is being used by a wrapper to also build a
        DataFrame, in such case the "Build DF" string will be added to the tdqm text
        :return: A Generator of FHIR bundles containing the queried information
        """
        hashed_request_param = None
        if self.bundle_cache_folder is not None:
            hashed_request_param = (
                self.bundle_cache_folder
                / hashlib.sha256(
                    (
                        self._build_request_url(
                            resource_type=resource_type,
                            request_params=request_params or {},
                        )
                        + ".json"
                    ).encode()
                ).hexdigest()
            )
        if (
            read_from_cache
            and hashed_request_param is not None
            and hashed_request_param.exists()
        ):
            logger.info(f"Reading data from {hashed_request_param}")
            with hashed_request_param.open("r") as fp:
                n_bundles = 0
                for b in json.load(fp=fp):
                    yield FHIRObj(**b)
                    n_bundles += 1
                return n_bundles
        else:
            bundles = [
                bundle
                for bundle in self._get_bundles(
                    resource_type=resource_type,
                    request_params=request_params,
                    num_pages=num_pages,
                    silence_tqdm=silence_tqdm,
                    tqdm_df_build=tqdm_df_build,
                )
            ]
        if hashed_request_param is not None:
            logger.info(f"Storing data to {hashed_request_param}")
            # TODO: Try this with big files
            # Store the bundles
            with hashed_request_param.open("w") as fp:
                fp.write("[" + ",".join(b.to_json() for b in bundles) + "]")
        for bundle in bundles:
            yield bundle
        return len(bundles)

    def _bundle_fn(self, *args: Any, **kwargs: Any) -> Generator[FHIRObj, None, int]:
        """
        Wrapper function runs either `_get_bundles` or `_cache_bundle`, depending on whether a
        caching folder has been defined.
        """
        if self.bundle_cache_folder is not None:
            fn = self._cache_bundles
        else:
            fn = self._get_bundles
        return fn(*args, **kwargs)

    def _bundle_fn_to_list(self, *args: Any, **kwargs: Any) -> List[FHIRObj]:
        """
        Wrapper function that converts the result of `_bundle_fn` (either `_get_bundles` or
        `_cache_bundle`, depending on whether a caching folder has been defined) to a list.
        """
        return list(self._bundle_fn(*args, **kwargs))

    def _get_bundles_for_timespan(
        self,
        resource_type: str,
        request_params: Dict[str, Any],
        time_attribute_name: str,
        timespan: Tuple[str, str],
        *args: Any,
        **kwargs: Any,
    ) -> Generator[FHIRObj, None, int]:
        """
        Wrapper function that sets the `time_attribute_name` date parameters for the
        `sail_through_search_space function`.
        """
        request_params[time_attribute_name] = (
            f"ge{timespan[0]}",
            f"lt{timespan[1]}",
        )
        return self._bundle_fn(
            resource_type=resource_type,
            request_params=request_params,
            *args,
            **kwargs,
        )

    def _bundles_for_timespan_to_list(
        self,
        resource_type: str,
        request_params: Dict[str, Any],
        time_attribute_name: str,
        timespan: Tuple[str, str],
        *args: Any,
        **kwargs: Any,
    ) -> List[FHIRObj]:
        """
        Wrapper function that converts the result of `_bundles_for_timespan_to_list` to a list.
        """
        return list(
            self._get_bundles_for_timespan(
                resource_type,
                request_params,
                time_attribute_name,
                timespan,
                *args,
                **kwargs,
            )
        )

    ##############################
    #      QUERY HANDLING        #
    ##############################

    def _run_multiquery(
        self,
        func: Callable,
        params: List[Any],
        tqdm_text: str,
        disable_multiprocess: bool = False,
    ) -> Generator[FHIRObj, None, int]:
        n_bundles = 0
        if disable_multiprocess:
            for param in tqdm(params, total=len(params), desc=tqdm_text):
                for bundle in func(param):
                    yield bundle
                    n_bundles += 1
        else:
            pool = multiprocessing.Pool(processes=self.num_processes)
            for bundles_per_query in tqdm(
                pool.imap(func, params), total=len(params), desc=tqdm_text
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
        read_from_cache: bool = False,
        disable_multiprocessing: bool = False,
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
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
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
                logger.warning(
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
            func = partial(
                # If multiprocessing is disabled, I can use the generators,
                # otherwise I have to use a list
                self._get_bundles_for_timespan
                if disable_multiprocessing
                else self._bundles_for_timespan_to_list,
                resource_type,
                request_params,
                time_attribute_name,
                num_pages=-1,
                silence_tqdm=True,
                read_from_cache=read_from_cache,
            )
            return self._run_multiquery(
                func=func,
                params=timespans,
                tqdm_text=f"Query Timespans & Build DF ({resource_type})"
                if tqdm_df_build
                else f"Query Timespans ({resource_type})",
                disable_multiprocess=disable_multiprocessing,
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
        read_from_cache: bool = False,
        disable_multiprocessing: bool = False,
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
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :param disable_multiprocessing: If true, the bundles will be processed sequentially and
        then returned as a Generator; if false, the bundles for each query will be first converted
        to a list (because Generators cannot be pickled, and thus cannot be used by multiprocessing),
        and then they will be yielded as a Generator
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
            func = partial(
                # If multiprocessing is disabled, I can use the generators,
                # otherwise I have to use a list
                self._bundle_fn if disable_multiprocessing else self._bundle_fn_to_list,
                resource_type,
                num_pages=num_pages,
                read_from_cache=read_from_cache,
                silence_tqdm=True,
            )
            return self._run_multiquery(
                func=func,
                params=request_params_per_sample,
                tqdm_text=f"Query Rows & Build DF ({resource_type})"
                if tqdm_df_build
                else f"Query Rows ({resource_type})",
                disable_multiprocess=disable_multiprocessing,
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
                logger.warning(
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
                        logger.warning(
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
        merge_on: str = None,
        build_df_after_query: bool = False,
        disable_multiprocessing: bool = False,
    ) -> pd.DataFrame:
        """
        Convert a bundle into a DataFrame using either the `flatten_data` function (default),
        FHIR paths or a custom processing function. For the case of `flatten_data` and the FHIR
        paths, each row of the DataFrame will represent a resource. In the custom processing
        function each bundle can be handled as one pleases.

        :param bundles: The bundles to transform
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param merge_on: Whether to merge the results on a certain row after computing. This is
        useful when using includes, if you store the IDs on the same column you can use that column
        to merge all the rows into one, an example is given in `bundles_to_dataframe`
        :param build_df_after_query: Whether the DataFrame should be built after all bundles have
        been collected, or whether the bundles should be transformed just after retrieving
        :param disable_multiprocessing: Whether the bundles should be processed sequentially
        :return: A pandas DataFrame containing the queried information
        """
        if disable_multiprocessing:
            results = [item for bundle in bundles for item in process_function(bundle)]
        else:
            pool = multiprocessing.Pool(self.num_processes)
            if build_df_after_query or isinstance(bundles, List):
                bundles = list(bundles)
                results = [
                    item
                    for sublist in tqdm(
                        pool.imap(process_function, bundles),
                        total=len(bundles),
                        desc="Build DF",
                    )
                    for item in sublist
                ]
            else:
                results = [
                    item
                    for sublist in pool.imap(process_function, bundles)
                    for item in sublist
                ]
            pool.close()
            pool.join()
        df = pd.DataFrame(results)
        return (
            df if merge_on is None or len(df) == 0 else self.merge_on_col(df, merge_on)
        )

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
            merge_on: str = None,
            build_df_after_query: bool = False,
            disable_post_multiprocessing: bool = False,
            *args: Any,
            **kwargs: Any,
        ) -> pd.DataFrame:
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
                    merge_on=merge_on,
                    build_df_after_query=build_df_after_query,
                    disable_multiprocessing=disable_post_multiprocessing,
                )

        return wrap

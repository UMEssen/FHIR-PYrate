import datetime
import hashlib
import json
import logging
import math
import multiprocessing
import re
import traceback
import warnings
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import requests
from dateutil.parser import parse
from tqdm import tqdm

from fhir_pyrate.util import FHIRObj, string_from_column
from fhir_pyrate.util.bundle_processing_templates import flatten_data, parse_fhir_path


class Pirate:
    """
    Main class to query resources using the FHIR API.

    :param base_url: The main URL where the FHIR server is located
    :param auth: An authentication service that can be used obtain a session for querying resources
    :param num_processes: The number of processes that should be used to run the query for the
    functions that use multiprocessing
    :param print_request_url: Whether the request URLs should be printed whenever we do a request
    :param time_format: The time format used by the FHIR API
    :param default_count: The default count of results per page used by the server
    :param bundle_cache_folder: Whether bundles should be stored for later use, and where
    :param silence_fhirpath_warning: Whether the FHIR path warning regarding already existing
    expressions should be silenced
    :param session: An already authenticated request.Session that can be used to run the FHIR
    queries.
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
    FHIRPATH_IMPORT_ERROR = (
        "The fhirpath-py package cannot be pushed as a package dependency because it is not "
        "present on PyPi, if you want to use the FHIRPath functionalities you need to install "
        "this package using "
        "`pip install git+https://github.com/beda-software/fhirpath-py.git` or by adding it to "
        "poetry."
    )

    def __init__(
        self,
        base_url: str,
        auth: Any,
        num_processes: int = 1,
        print_request_url: bool = False,
        time_format: str = "%Y-%m-%dT%H:%M",
        default_count: int = None,
        bundle_cache_folder: Union[str, Path] = None,
        silence_fhirpath_warning: bool = False,
        session: requests.Session = None,
        optional_get_params: Dict[Any, Any] = None,
    ):
        # Remove the last character if they added it
        url_search = re.search(
            pattern=r"(https?:\/\/[^\/]+)([\w\.\-~\/]*)", string=base_url
        )
        if url_search is None:
            raise ValueError(
                "The given URL does not follow the validation RegEx. Was your URL "
                "written correctly? If it is, please create an issue."
            )
        self.base_url = url_search.group(1)
        self.fhir_app_location = (
            url_search.group(2)
            if len(url_search.group(2)) > 0 and url_search.group(2)[-1] == "/"
            else url_search.group(2) + "/"
        )
        self.auth = auth
        self._close_session_on_exit = False
        if session is not None:
            self.session = session
        elif self.auth is not None:
            self.session = self.auth.session
        else:
            self.session = requests.Session()
            self._close_session_on_exit = True
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
            logging.warning(
                "Bundle caching is a beta feature. This has not yet been extensively "
                "tested and does not have any cache invalidation mechanism."
            )
            self.bundle_cache_folder = Path(bundle_cache_folder)
            self.bundle_cache_folder.mkdir(parents=True, exist_ok=True)

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

    @staticmethod
    def _concat_request_params(request_params: Dict[str, Any]) -> str:
        """
        Concatenates the parameters to create a request string.

        :param request_params: The parameters that should be used for the request
        :return: The concatenated string for the request
        """
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

    def _get_response(self, request_url: str) -> Optional[FHIRObj]:
        """
        Performs the API request and returns the response as a dictionary.

        :param request_url: The request string
        :return: A FHIR bundle
        """
        try:
            response = self.session.get(request_url, **self.optional_get_params)
            if self._print_request_url:
                print(request_url)
            response.raise_for_status()
            json_response = FHIRObj(**response.json())
            return json_response
        except Exception:
            # Leave this to be able to quickly see the errors
            logging.error(traceback.format_exc())
            return None

    def steal_bundles(
        self,
        resource_type: str,
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
        read_from_cache: bool = False,
        silence_tqdm: bool = False,
    ) -> List[FHIRObj]:
        """
        Wrapper for the steal_bundles_for_timespan function to disable the use of a timestamp.

        :param resource_type: The resource to be queried, e.g. DiagnosticReport
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :param silence_tqdm: Whether tqdm should be disabled
        :return: A list of bundles with the queried information
        """
        return self.steal_bundles_for_timespan(
            request_params=request_params,
            resource_type=resource_type,
            num_pages=num_pages,
            read_from_cache=read_from_cache,
            silence_tqdm=silence_tqdm,
        )

    def steal_bundles_for_timespan(
        self,
        resource_type: str,
        request_params: Dict[str, Any] = None,
        time_attribute_name: str = "_lastUpdated",
        time_interval: Tuple[str, str] = None,
        num_pages: int = -1,
        read_from_cache: bool = False,
        silence_tqdm: bool = False,
    ) -> List[FHIRObj]:
        """
        Executes a request and iterates through the results pages and stores all bundles in a
        list. If bundle caching is activated and read_from_cache is true, then the bundles will
        be read from file instead.

        :param resource_type: The resource to be queried, e.g. DiagnosticReport
        :param request_params: The parameters for the query, e.g. _count, _id
        :param time_attribute_name: The time attribute that should be used to define the
        timespan; e.g. started for ImagingStudy, date for DiagnosticReport. The default value is
        _lastUpdated, because it exists in all resources
        :param time_interval: The time interval for the query
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :param silence_tqdm: Whether tqdm should be disabled
        :return: A list of bundles with the queried information
        """
        bundles: List[FHIRObj] = []
        if num_pages == 0:
            return bundles
        current_params = {} if request_params is None else request_params.copy()
        if time_interval is not None:
            current_params[time_attribute_name] = (
                f"ge{time_interval[0]}",
                f"lt{time_interval[1]}",
            )

        request_params_string = self._concat_request_params(current_params)
        hashed_request_param = None
        if self.bundle_cache_folder is not None:
            hashed_request_param = (
                self.bundle_cache_folder
                / hashlib.sha256(
                    (f"{resource_type}?{request_params_string}" + ".json").encode()
                ).hexdigest()
            )
            if read_from_cache and hashed_request_param.exists():
                with hashed_request_param.open("r") as fp:
                    bundles = [FHIRObj(**b) for b in json.load(fp=fp)]
                    assert isinstance(bundles, list)
                    return bundles

        bundle = self._get_response(
            f"{self.base_url}{self.fhir_app_location}{resource_type}?{request_params_string}"
        )
        if time_interval is None:
            self._check_sorting(bundle, current_params)

        bundle_total: Union[int, float] = num_pages
        if bundle_total == -1:
            total = self._get_total_from_bundle(bundle, count_entries=False)
            n_entries = self._get_total_from_bundle(bundle, count_entries=True)
            if total and n_entries:
                bundle_total = math.ceil(total / n_entries)
            else:
                bundle_total = math.inf
        progress_bar = tqdm(disable=silence_tqdm, desc="Query", total=bundle_total)
        while bundle is not None:
            progress_bar.update()
            if time_interval is not None:
                total = self._get_total_from_bundle(bundle, count_entries=True)
                if total is None or total == 0:
                    warnings.warn(
                        f"The bundle retrieved for the time between {time_interval[0]} and "
                        f"{time_interval[1]} is empty. Do you want to choose another time "
                        f"frame?"
                    )
            # Add to the list
            bundles.append(bundle)
            # Find the next page, if it exists
            next_link_url = next(
                (link.url for link in bundle.link or [] if link.relation == "next"),
                None,
            )
            if next_link_url is None or progress_bar.n >= bundle_total:
                break
            else:
                # Re-assign bundle and start new iteration
                bundle = self._get_response(
                    f"{self.base_url}{next_link_url}"
                    if self.base_url not in next_link_url
                    else next_link_url  # on HAPI the server
                )
        progress_bar.close()
        if hashed_request_param is not None:
            # TODO: Try this with big files
            # Store the bundles
            with hashed_request_param.open("w") as fp:
                fp.write("[" + ",".join(b.to_json() for b in bundles) + "]")
        return bundles

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

    def _check_sorting(
        self,
        bundle: Optional[FHIRObj],
        request_params: Dict[str, Any],
    ) -> None:
        """
        Perform some sanity checks on the request parameters.

        :param bundle: An initial bundle with count=0 that is downloaded to check the amount of
        requests that will need to be done later
        :param request_params: The parameters for the current request
        """
        given_count = self._return_count_from_request(request_params)
        total = self._get_total_from_bundle(bundle, count_entries=True)
        if given_count is None or total is None:
            return
        if (
            bundle is not None
            and total > given_count
            and not any(k == "_sort" for k, _ in request_params.items())
        ):
            warnings.warn(
                f"The bundle has multiple pages (_count = {given_count}, "
                f"results = {total}) but no sorting method has been defined, "
                "which may yield incorrect results. We will set the sorting parameter to id."
            )
            request_params["_sort"] = "_id"

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
        request_params = {} if request_params is None else request_params
        request_params_string = self._concat_request_params(request_params)
        request_url = (
            f"{self.base_url}{self.fhir_app_location}{resource_type}"
            f"?{request_params_string}"
        )
        return self._get_total_from_bundle(
            bundle=self._get_response(request_url), count_entries=count_entries
        )

    def sail_through_search_space(
        self,
        resource_type: str,
        time_attribute_name: str,
        date_init: Union[str, datetime.date],
        date_end: Union[str, datetime.date],
        request_params: Dict[str, Any] = None,
        read_from_cache: bool = False,
    ) -> List[FHIRObj]:
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
        :return: A FHIR bundle containing the queried information for all timespans
        """
        # Check if the date parameters that we use for multiprocessing are already included in
        # the request parameters
        request_params = {} if request_params is None else request_params
        search_division_params = [
            k for k in request_params.keys() if k == time_attribute_name
        ]
        # If they are, remove them and issue a warning
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
        request_params_with_date = request_params.copy()
        # Add count = 0 to not return any entries
        request_params_with_date["_count"] = 0
        request_params_with_date[time_attribute_name] = (
            f"ge{date_init}",
            f"lt{date_end}",
        )
        request_params_string = self._concat_request_params(request_params_with_date)
        request_url = (
            f"{self.base_url}{self.fhir_app_location}{resource_type}"
            f"?{request_params_string}"
        )
        bundle = self._get_response(request_url)
        self._check_sorting(bundle, request_params)
        logging.info(
            f"Running sail_through_search_space with {self.num_processes} processes."
        )
        # Divide the current time period into smaller spans
        timespans = self._get_timespan_list(date_init, date_end)
        pool = multiprocessing.Pool(processes=self.num_processes)
        func = partial(
            self.steal_bundles_for_timespan,
            resource_type,
            request_params,
            time_attribute_name,
            num_pages=-1,
            read_from_cache=read_from_cache,
            silence_tqdm=True,
        )
        bundles = [
            item
            for sublist in tqdm(
                pool.imap(func, timespans), total=len(timespans), desc="Query"
            )
            for item in sublist
        ]
        pool.close()
        pool.join()
        return bundles

    @staticmethod
    def _get_request_params_for_sample(
        df: pd.DataFrame,
        request_params: Dict[str, Any],
        df_constraints: Dict[str, Union[str, Tuple[str, str]]],
    ) -> List[Dict[str, str]]:
        return [
            dict(
                {
                    fhir_identifier: (
                        row[df.columns.get_loc(second_term)]
                        if isinstance(second_term, str)
                        else str(second_term[0])
                        + str(row[df.columns.get_loc(second_term[1])])
                    )
                    # Concatenate the given system identifier string with the desired identifier
                    for fhir_identifier, second_term in df_constraints.items()
                },
                **request_params,
            )
            for row in df.itertuples(index=False)
        ]

    def trade_rows_for_bundles(
        self,
        df: pd.DataFrame,
        resource_type: str,
        df_constraints: Dict[str, Union[str, Tuple[str, str]]],
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
        read_from_cache: bool = False,
    ) -> List[FHIRObj]:
        """
        Go through the rows of a DataFrame (with multiprocessing) and run a query and retrieve
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
        :return: A FHIR bundle containing the queried information
        """
        request_params = {} if request_params is None else request_params
        logging.info(
            f"Querying each row of the DataFrame with {self.num_processes} processes."
        )
        request_params_per_sample = self._get_request_params_for_sample(
            df=df, request_params=request_params, df_constraints=df_constraints
        )
        pool = multiprocessing.Pool(self.num_processes)
        func = partial(
            self.steal_bundles,
            resource_type,
            num_pages=num_pages,
            read_from_cache=read_from_cache,
            silence_tqdm=True,
        )
        bundles = [
            item
            for sublist in tqdm(
                pool.imap(func, request_params_per_sample),
                total=len(request_params_per_sample),
                desc="Query",
            )
            for item in sublist
        ]
        pool.close()
        pool.join()

        return bundles

    def bundles_to_dataframe(
        self,
        bundles: List[FHIRObj],
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
        sequential_df_build: bool = False,
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
        :param sequential_df_build: This variable is set to true by other functions in the Pirate
        class whenever the creation of a DataFrame happens directly after the query
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
        df = search.query_to_dataframe(
            bundles_function=search.steal_bundles,
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
        """
        if fhir_paths is not None:
            try:
                from fhirpathpy import compile
            except ImportError as e:
                raise ImportError(self.FHIRPATH_IMPORT_ERROR) from e
            logging.debug(
                f"The selected process_function {process_function.__name__} will be "
                f"overwritten."
            )
            fhir_paths_with_name = [
                (path[0], path[1]) if isinstance(path, tuple) else (path, path)
                for path in fhir_paths
            ]
            if not self.silence_fhirpath_warning:
                for _, path in fhir_paths_with_name:
                    for token in self.FHIRPATH_INVALID_TOKENS:
                        if (
                            re.search(
                                pattern=rf"{token}[\.\[]|[\.\]]{token}$", string=path
                            )
                            is not None
                        ):
                            logging.warning(
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
                (name, compile(path=path)) for name, path in fhir_paths_with_name
            ]
            process_function = partial(
                parse_fhir_path, compiled_fhir_paths=fhir_paths_with_name
            )
        if self.num_processes > 1 and not sequential_df_build:
            pool = multiprocessing.Pool(self.num_processes)
            results = [
                item
                for sublist in tqdm(
                    pool.imap(process_function, bundles),
                    total=len(bundles),
                    desc="Build DF",
                )
                for item in sublist
            ]
            pool.close()
            pool.join()
        else:
            results = [
                item
                for bundle in tqdm(
                    bundles,
                    disable=sequential_df_build,
                    total=len(bundles),
                    desc="Build DF",
                )
                for item in process_function(bundle)
            ]
        return pd.DataFrame(results)

    def trade_rows_for_dataframe(
        self,
        df: pd.DataFrame,
        resource_type: str,
        df_constraints: Dict[str, Union[str, Tuple[str, str]]],
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
        request_params: Dict[str, Any] = None,
        num_pages: int = -1,
        read_from_cache: bool = False,
    ) -> pd.DataFrame:
        """
        Go through the rows of a DataFrame (with multiprocessing) and run a query and retrieve
        bundles and store them in a DataFrame. There are two differences between this approach
        and trade_rows_for_bundles:
        1. Here, the bundles are retrieved and the DataFrame is computed straight away. In
        query_to_dataframe(bundles_function=self.trade_rows_for_bundles, ...) first all the
        bundles are retrieved, and then they are converted into a DataFrame.
        2. If the df_constraints constraints are specified, they will end up in the final DataFrame.

        :param df: The DataFrame with the queries
        :param resource_type: The resource to query, e.g. Patient, DiagnosticReport
        :param df_constraints: A dictionary containing a mapping between the FHIR attributes and
        the columns of the input DataFrame, e.g. {"subject" : "fhir_patient_id"}, where subject
        is the FHIR attribute and fhir_patient_id is the name of the column. It is also possible
        to add the system by using a tuple instead of a string, e.g. "code": (
        "http://loinc.org", "loinc_code")
        :param process_function: The transformation function going through the entries and
        storing the entries to save
        :param fhir_paths: A list of FHIR paths (https://hl7.org/fhirpath/) to be used to build the
        DataFrame, alternatively, a list of tuples can be used to specify the column name of the
        future column with (column_name, fhir_path). Please refer to the `bundles_to_dataframe`
        functions for notes on how to use the FHIR paths.
        :param request_params: The parameters for the query, e.g. _count, _id
        :param num_pages: The number of pages of bundles that should be returned, the default is
        -1 (all bundles), with any other value exactly that value of bundles will be returned,
        assuming that there are that many
        :param read_from_cache: Whether we should read the bundles from a cache folder,
        in case they have already been computed
        :return: A pandas DataFrame containing the queried information merged with the original
        DataFrame
        """
        if fhir_paths is not None:
            try:
                import fhirpathpy  # noqa
            except ImportError as e:
                raise ImportError(self.FHIRPATH_IMPORT_ERROR) from e
        request_params = {} if request_params is None else request_params
        logging.info(
            f"Querying each row of the DataFrame with {self.num_processes} processes."
        )
        req_params_per_sample = self._get_request_params_for_sample(
            df=df, request_params=request_params, df_constraints=df_constraints
        )
        input_params_per_sample = [
            {
                # The name of the parameter will be the same as the column name
                # The value will be the same as the value in that column for that row
                (second_term if isinstance(second_term, str) else second_term[1]): (
                    row[df.columns.get_loc(second_term)]
                    if isinstance(second_term, str)
                    else str(row[df.columns.get_loc(second_term[1])])
                )
                # Concatenate the given system identifier string with the desired identifier
                for _, second_term in df_constraints.items()
            }
            for row in df.itertuples(index=False)
        ]
        # Add all the parameters needed by the apply_async function
        params_per_sample = [
            {
                "resource_type": resource_type,
                "request_params": req_sample,
                "bundles_function": self.steal_bundles,
                "process_function": process_function,
                "fhir_paths": fhir_paths,
                "num_pages": num_pages,
                "read_from_cache": read_from_cache,
                "silence_tqdm": True,
                "sequential_df_build": True,
            }
            for req_sample in req_params_per_sample
        ]
        found_dfs = []
        # TODO: Too late for this.
        #  Refactor this into one single function, the problem is currently that the
        #  notebooks are not happy with multiprocessing
        if self.num_processes > 1:
            pool = multiprocessing.Pool(self.num_processes)
            results = []
            for param, input_param in zip(params_per_sample, input_params_per_sample):
                results.append(
                    (
                        pool.apply_async(self.query_to_dataframe, kwds=param),
                        input_param,
                    )
                )
            for async_result, input_param in tqdm(
                results, total=len(results), desc="Query & Build DF"
            ):
                found_df = async_result.get()
                for key, value in input_param.items():
                    found_df[key] = value
                found_dfs.append(found_df)
            pool.close()
            pool.join()
        else:
            for param, input_param in tqdm(
                zip(params_per_sample, input_params_per_sample),
                total=len(params_per_sample),
                desc="Query & Build DF",
            ):
                found_df = self.query_to_dataframe(**param)  # type: ignore
                for key, value in input_param.items():
                    found_df[key] = value
                found_dfs.append(found_df)
        return pd.concat(found_dfs, ignore_index=True)

    def query_to_dataframe(
        self,
        bundles_function: Callable,
        process_function: Callable[[FHIRObj], Any] = flatten_data,
        fhir_paths: List[Union[str, Tuple[str, str]]] = None,
        sequential_df_build: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
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
        :param sequential_df_build: This variable is set to true by other functions in the Pirate
        class whenever the creation of a DataFrame happens directly after the query
        :param kwargs: The arguments that will be passed to the `bundles_function` function,
        please refer to the documentation of the respective methods.
        :return: A pandas DataFrame containing the queried information
        """
        if fhir_paths is not None:
            try:
                import fhirpathpy  # noqa
            except ImportError as e:
                raise ImportError(self.FHIRPATH_IMPORT_ERROR) from e
        return self.bundles_to_dataframe(
            bundles=bundles_function(**kwargs),
            process_function=process_function,
            fhir_paths=fhir_paths,
            sequential_df_build=sequential_df_build,
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

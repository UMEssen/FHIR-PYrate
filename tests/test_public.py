import logging
import os
import re
import unittest
from typing import Dict, List

import pandas as pd
from bs4 import BeautifulSoup

from fhir_pyrate import Ahoy, Miner, Pirate
from fhir_pyrate.util import FHIRObj

logging.getLogger().setLevel(logging.INFO)

SERVERS = [
    ("http://hapi.fhir.org/baseDstu2", "patient"),
    ("http://hapi.fhir.org/baseDstu3", "subject"),
    ("http://hapi.fhir.org/baseR4", "subject"),
    ("http://hapi.fhir.org/baseR5", "subject"),
    # ("https://stu3.test.pyrohealth.net/fhir", "subject"),
]

AUTH_SERVERS = [
    ("http://hapi.fhir.org/baseDstu2", None, None),
    # Give username and password don't work
    # ("https://jade.phast.fr/resources-server/api/FHIR/", "basicauth", "env"),
]

TEST_URLS = [
    "http://hapi.fhir.org/baseDstu2/a/b/c",
    "http://hapi.fhir.org/baseDstu2/a/b/c/",
    "http://hapi.fhir.org/baseDstu2",
    "http://hapi.fhir.org/baseDstu2/",
    "https://hapi.fhir.org/baseDstu2",
    "https://hapi.fhir.org/",
    "https://hapi.fhir.org",
]
NEXT_LINK_URLS = [
    "/baseDstu2?_getpages=9f39b4db-cd37-4fdc-b43e-d790d18f0778&_getpagesoffset=10&_count=10&_pretty=true&_bundletype=searchset",
    "https://hapi.fhir.org/baseDstu2?_getpages=9f39b4db-cd37-4fdc-b43e-d790d18f0778&_getpagesoffset=10&_count=10&_pretty=true&_bundletype=searchset",
]


# Processing function to process each single text
def decode_text(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    div = soup.find("div", {"class": "hapiHeaderText"})
    return str(div.text)


def get_diagnostic_text(bundle: FHIRObj) -> List[Dict]:
    records = []
    for entry in bundle.entry or []:
        resource = entry.resource
        records.append(
            {
                "fhir_diagnostic_report_id": resource.id,
                "report_status": resource.text.status
                if resource.text is not None
                else None,
                "report_text": resource.text.div if resource.text is not None else None,
            }
        )
    return records


def get_observation_info(bundle: FHIRObj) -> List[Dict]:
    records = []
    for entry in bundle.entry or []:
        resource = entry.resource
        # Store the ID
        base_dict = {"observation_id": resource.id}
        for component in resource.component or []:
            # Go through the code.codings of the current components to get a name for our value
            # and store the display value
            resource_name = next(
                iter([coding.display for coding in component.code.coding or []]), ""
            )
            if component.valueQuantity is not None:
                # If the component is a valueQuantity, get the value
                base_dict[resource_name] = component.valueQuantity.value
                base_dict[resource_name + " Unit"] = component.valueQuantity.unit
        records.append(base_dict)
    return records


class GeneralTests(unittest.TestCase):
    def testURL(self) -> None:
        for test_url in TEST_URLS:
            for next_link_url in NEXT_LINK_URLS:
                url_search = re.search(
                    pattern=r"(https?:\/\/([^\/]+))([\w\.\-~\/]*)", string=test_url
                )
                assert url_search is not None
                base_url = url_search.group(1)
                domain = url_search.group(2)
                fhir_app_location = (
                    url_search.group(3)
                    if len(url_search.group(3)) > 0 and url_search.group(3)[-1] == "/"
                    else url_search.group(3) + "/"
                )
                assert domain in base_url
                build_link = f"{base_url}{fhir_app_location}DiagnosticReport"
                assert "/DiagnosticReport" in build_link, build_link
                assert build_link.count("http") == 1, build_link
                next_link = (
                    f"{base_url}{next_link_url}"
                    if domain not in next_link_url
                    else next_link_url
                )
                assert next_link.count("http") == 1, next_link

    def testAuth(self) -> None:
        for server, type, method in AUTH_SERVERS:
            if "jade.phast.fr" in server:
                # TODO: Does anybody know a reliable BasicAuth test server?
                #  This one does not work.
                os.environ["FHIR_USER"] = "Connectathon"
                os.environ["FHIR_PASSWORD"] = "Connectathon_052020"
            with self.subTest(msg="{}".format(server)):
                with Ahoy(
                    auth_type=type,
                    auth_url=server,
                    auth_method=method,
                ) as auth:
                    assert auth.session is not None
                    if "jade.phast.fr" in server:
                        assert auth.session.auth is not None
                    search = Pirate(
                        auth=auth,
                        base_url=server,
                        print_request_url=False,
                        num_processes=2,
                    )
                    value_df = search.steal_bundles_to_dataframe(
                        resource_type="ValueSet",
                        num_pages=1,
                        request_params={"_sort": "_id"},
                    )
                    assert len(value_df) > 0

    # Test each single function from Pirate
    def testServers(self) -> None:
        for server, patient_ref in SERVERS:
            with self.subTest(msg="{}".format(server)):
                with Pirate(
                    auth=None,
                    base_url=server,
                    print_request_url=False,
                    num_processes=2,
                ) as search:
                    if server == "https://stu3.test.pyrohealth.net/fhir":
                        condition_df = search.steal_bundles_to_dataframe(
                            resource_type="Condition",
                            request_params={
                                "_format": "json",
                            },
                            # some servers have patient, some have subject
                            fhir_paths=["id", ("patient", f"{patient_ref}.reference")],
                        )
                    else:
                        condition_df = search.sail_through_search_space_to_dataframe(
                            resource_type="Condition",
                            request_params={
                                "_count": 100,
                                "_sort": "_id",
                                "_format": "json",
                            },
                            time_attribute_name="_lastUpdated",
                            date_init="2021-01-01",
                            date_end="2022-01-01",
                            # some servers have patient, some have subject
                            fhir_paths=[
                                ("condition", "id"),
                                ("patient", f"{patient_ref}.reference"),
                            ],
                        )
                    condition_df.dropna(axis=0, inplace=True, how="any")
                    assert len(condition_df) > 0
                    diagnostic_df = search.trade_rows_for_dataframe(
                        df=condition_df.head(1),
                        resource_type="DiagnosticReport",
                        request_params={
                            "_format": "json",
                        },
                        df_constraints={
                            patient_ref: "patient",
                        },
                        process_function=get_diagnostic_text,
                    )
                    if len(diagnostic_df) > 0:
                        diagnostic_df.dropna(
                            subset=["report_text"],
                            inplace=True,
                        )  # Remove the rows with invalid values
                    if len(diagnostic_df) > 0:
                        miner = Miner(target_regex="Panel*")

                        df_filtered = miner.nlp_on_dataframe(
                            diagnostic_df,
                            text_column_name="report_text",
                            new_column_name="text_found",
                        )
                        assert df_filtered["text_found"].sum() > 0


class ExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.search = Pirate(
            auth=None,
            base_url="http://hapi.fhir.org/baseDstu2",
            print_request_url=False,
            num_processes=1,
        )
        super().setUp()

    def tearDown(self) -> None:
        self.search.close()

    def testExample1(self) -> None:
        observation_all = self.search.steal_bundles_to_dataframe(
            resource_type="Observation",
            request_params={
                "_id": "86092",
            },
        )
        assert len(observation_all) == 1
        observation_values = self.search.steal_bundles_to_dataframe(
            resource_type="Observation",
            request_params={
                "_count": 1,
                "_id": "86092",
            },
            fhir_paths=[
                "id",
                "effectiveDateTime",
                ("value", "valueQuantity.value"),
                ("unit", "valueQuantity.unit"),
                ("patient", "subject.reference.replace('Patient/', ''"),
            ],
        )
        assert len(observation_values) == 1
        assert (
            observation_values.iloc[0, 2] == 6.079781499882176
        ), observation_values.iloc[0, 2]

    def testExample3(self) -> None:
        condition_df = self.search.steal_bundles_to_dataframe(
            resource_type="Condition",
            request_params={
                "_count": 100,
                "code": "http://snomed.info/sct%7C84757009",
                "_sort": "_id",
            },
            fhir_paths=[
                "id",
                ("patient_id", "patient.reference"),
                "verificationStatus",
            ],
        )
        assert len(condition_df) == 31
        patient_df = condition_df["patient_id"].drop_duplicates(keep="first").to_frame()
        assert len(patient_df) == 31
        observation_df = self.search.trade_rows_for_dataframe(
            df=patient_df,
            resource_type="Observation",
            request_params={
                "_count": 100,
                "code": "http://loinc.org%7C55284-4",
                "_sort": "_id",
            },
            with_ref=True,
            df_constraints={"subject": "patient_id"},
            fhir_paths=[
                "id",
                "effectiveDateTime",
                ("test", "component.code.coding.display"),
                ("value", "component.valueQuantity.value"),
                ("unit", "component.valueQuantity.unit"),
            ],
        ).explode(
            [
                "test",
                "value",
                "unit",
            ]
        )
        assert len(observation_df) == 120 * 2
        observation_df = self.search.trade_rows_for_dataframe(
            df=patient_df,
            resource_type="Observation",
            request_params={
                "_count": 100,
                "code": "http://loinc.org%7C55284-4",
                "_sort": "_id",
            },
            with_ref=True,
            df_constraints={"subject": "patient_id"},
            process_function=get_observation_info,
        )
        assert len(observation_df) == 120

    def testExample4Exception(self) -> None:
        self.assertRaises(
            Exception,
            self.search.steal_bundles_to_dataframe,
            resource_type="DiagnosticReport",
            request_params={
                "_count": 100,
                "_lastUpdated": "ge2021",
            },
            fhir_paths=["text.status", "text.div"],
        )

    def testExample4(self) -> None:
        diagnostic_df = self.search.steal_bundles_to_dataframe(
            resource_type="DiagnosticReport",
            request_params={
                "_count": 100,
                "_lastUpdated": "ge2021",
            },
            process_function=get_diagnostic_text,  # Use processing function
        )
        assert len(diagnostic_df) > 47

        miner = Miner(
            target_regex="Metabolic", decode_text=decode_text, num_processes=1
        )
        df_filtered = miner.nlp_on_dataframe(
            diagnostic_df,
            text_column_name="report_text",
            new_column_name="text_found",
        )
        assert sum(df_filtered["text_found"]) == 33


class TestPirate(unittest.TestCase):
    def setUp(self) -> None:
        self.search = Pirate(
            auth=None,
            base_url="http://hapi.fhir.org/baseDstu2",
            print_request_url=False,
            num_processes=3,
        )
        super().setUp()

    def tearDown(self) -> None:
        self.search.close()

    def testStealBundles(self) -> None:
        obs_bundles = self.search.steal_bundles(
            resource_type="Observation", num_pages=5
        )
        obs_df = self.search.bundles_to_dataframe(obs_bundles)
        assert len(obs_df) > 0
        first_length = len(obs_df)
        for build_after_query in [True, False]:
            with self.subTest(msg="build_after_query_{}".format(build_after_query)):
                obs_df = self.search.steal_bundles_to_dataframe(
                    resource_type="Observation",
                    num_pages=5,
                    build_df_after_query=build_after_query,
                )
                assert len(obs_df) == first_length

    def testSail(self) -> None:
        for multi in [True, False]:
            obs_bundles = self.search.sail_through_search_space(
                resource_type="Observation",
                time_attribute_name="_lastUpdated",
                date_init="2021-01-01",
                date_end="2022-01-01",
                disable_multiprocessing=multi,
            )
            obs_df = self.search.bundles_to_dataframe(obs_bundles)
        assert len(obs_df) > 0
        first_length = len(obs_df)
        for multi in [True, False]:
            for build_after_query in [True, False]:
                with self.subTest(
                    msg="multi_{}_build_after_query_{}".format(multi, build_after_query)
                ):
                    obs_df = self.search.sail_through_search_space_to_dataframe(
                        resource_type="Observation",
                        time_attribute_name="_lastUpdated",
                        date_init="2021-01-01",
                        date_end="2022-01-01",
                        build_df_after_query=build_after_query,
                        disable_multiprocessing=multi,
                    )
                    assert len(obs_df) == first_length

    def testTrade(self) -> None:
        trade_df = pd.DataFrame(["18262-6", "2571-8"], columns=["code"])
        for multi in [True, False]:
            with self.subTest(msg="multi_{}".format(multi)):
                obs_bundles = self.search.trade_rows_for_bundles(
                    trade_df,
                    resource_type="Observation",
                    df_constraints={"code": "code"},
                    request_params={"_lastUpdated": "ge2020"},
                    disable_multiprocessing=multi,
                )
                obs_df = self.search.bundles_to_dataframe(obs_bundles)
                first_length = len(obs_df)
                assert len(obs_df) > 0

                obs_df = self.search.trade_rows_for_dataframe(
                    trade_df,
                    resource_type="Observation",
                    df_constraints={"code": "code"},
                    with_ref=True,
                    request_params={"_lastUpdated": "ge2020"},
                    disable_multiprocessing=multi,
                )
                assert len(obs_df) == first_length
        for multi in [True, False]:
            for build_after_query in [True, False]:
                with self.subTest(
                    msg="multi_{}_build_after_query_{}".format(multi, build_after_query)
                ):
                    obs_df = self.search.trade_rows_for_dataframe(
                        trade_df,
                        resource_type="Observation",
                        df_constraints={"code": "code"},
                        request_params={"_lastUpdated": "ge2020"},
                        build_df_after_query=build_after_query,
                        disable_multiprocessing=multi,
                    )
                    assert len(obs_df) == first_length


class ContraintsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.search = Pirate(
            auth=None,
            base_url="http://hapi.fhir.org/baseDstu2",
            print_request_url=False,
            num_processes=1,
        )
        super().setUp()
        self.condition_df = self.search.steal_bundles_to_dataframe(
            resource_type="Condition",
            request_params={
                "_count": 100,
                "_sort": "_id",
            },
            fhir_paths=[
                "id",
                ("patient_id", "patient.reference"),
                "verificationStatus",
                ("date", "onsetDateTime.substring(0,10)"),
            ],
            num_pages=1,
        )

    def tearDown(self) -> None:
        self.search.close()

    def testRefDiagnosticSubject(self) -> None:
        condition_df_pat = self.condition_df.loc[
            ~self.condition_df["patient_id"].isna()
        ]
        diagnostic_df = self.search.trade_rows_for_dataframe(
            df=condition_df_pat,
            resource_type="DiagnosticReport",
            df_constraints={
                "subject": "patient_id",
            },
            process_function=get_diagnostic_text,
        )
        assert len(diagnostic_df) > 0

    def testRefPatientSubject(self) -> None:
        condition_df_pat = self.condition_df.loc[
            ~self.condition_df["patient_id"].isna()
        ]
        patient_df = self.search.trade_rows_for_dataframe(
            df=condition_df_pat.head(10),
            resource_type="Patient",
            df_constraints={
                "_id": "patient_id",
            },
            fhir_paths=["id", "gender", "birthDate"],
        )
        assert len(patient_df) > 0

    def testRefDiagnosticDate(self) -> None:
        self.condition_df["date_end"] = "2022-01-01"
        condition_df_date = self.condition_df[~self.condition_df["date"].isna()]
        diagnostic_df = self.search.trade_rows_for_dataframe(
            df=condition_df_date.head(10),
            resource_type="DiagnosticReport",
            df_constraints={"date": [("ge", "date"), ("le", "date_end")]},
            fhir_paths=["id", "code.coding.code.display"],
            num_pages=2,
        )
        assert len(diagnostic_df) > 0

    def testRefEncounter(self) -> None:
        self.condition_df["code_column"] = "185345009"
        condition_df_date = self.condition_df[~self.condition_df["date"].isna()]
        encounter_df = self.search.trade_rows_for_dataframe(
            df=condition_df_date.head(10),
            resource_type="Encounter",
            df_constraints={"type": "code_column"},
            fhir_paths=["id", "class", "reason.coding.display"],
            num_pages=2,
        )
        assert len(encounter_df) > 0

    def testRefEncounterSystem(self) -> None:
        self.condition_df["code_column"] = "185345009"
        condition_df_date = self.condition_df[~self.condition_df["date"].isna()]
        encounter_df = self.search.trade_rows_for_dataframe(
            df=condition_df_date.head(10),
            resource_type="Encounter",
            df_constraints={"type": ("http://snomed.info/sct", "code_column")},
            fhir_paths=["id", "class", "reason.coding.display"],
            num_pages=2,
        )
        assert len(encounter_df) > 0


if __name__ == "__main__":
    unittest.main()

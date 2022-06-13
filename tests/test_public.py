import os
import unittest
from typing import Dict, List

from bs4 import BeautifulSoup

from fhir_pyrate import Ahoy, Miner, Pirate
from fhir_pyrate.util import FHIRObj

SERVERS = [
    ("http://hapi.fhir.org/baseDstu2", "patient"),
    ("http://hapi.fhir.org/baseDstu3", "subject"),
    ("http://hapi.fhir.org/baseR4", "subject"),
    ("http://hapi.fhir.org/baseR5", "subject"),
    ("https://stu3.test.pyrohealth.net/fhir", "subject"),
]

AUTH_SERVERS = [
    ("http://hapi.fhir.org/baseDstu2", None, None),
    # Give username and password don't work
    # ("https://jade.phast.fr/resources-server/api/FHIR/", "basicauth", "env"),
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
                    value_df = search.query_to_dataframe(
                        bundles_function=search.steal_bundles,
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
                        condition_df = search.query_to_dataframe(
                            bundles_function=search.steal_bundles,
                            resource_type="Condition",
                            request_params={
                                "_format": "json",
                            },
                            # some servers have patient, some have subject
                            fhir_paths=["id", ("patient", f"{patient_ref}.reference")],
                        )
                    else:
                        condition_df = search.query_to_dataframe(
                            bundles_function=search.sail_through_search_space,
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
        observation_all = self.search.query_to_dataframe(
            bundles_function=self.search.steal_bundles,
            resource_type="Observation",
            request_params={
                "_id": "86092",
            },
        )
        assert len(observation_all) == 1
        observation_values = self.search.query_to_dataframe(
            bundles_function=self.search.steal_bundles,
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
        condition_df = self.search.query_to_dataframe(
            bundles_function=self.search.steal_bundles,
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
            df_constraints={"subject": "patient_id"},
            process_function=get_observation_info,
        )
        assert len(observation_df) == 120

    def testExample4Exception(self) -> None:
        self.assertRaises(
            Exception,
            self.search.query_to_dataframe,
            bundles_function=self.search.steal_bundles,
            resource_type="DiagnosticReport",
            request_params={
                "_count": 100,
                "_lastUpdated": "ge2021",
            },
            fhir_paths=["text.status", "text.div"],
        )

    def testExample4(self) -> None:
        diagnostic_df = self.search.query_to_dataframe(
            bundles_function=self.search.steal_bundles,
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


if __name__ == "__main__":
    unittest.main()

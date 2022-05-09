import os
import unittest
from typing import Dict, List

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


class Test(unittest.TestCase):
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
                        stop_after_first_page=True,
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
                            fhir_paths=["id", f"{patient_ref}.reference"],
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
                            date_init="2020-01-01",
                            date_end="2022-01-01",
                            # some servers have patient, some have subject
                            fhir_paths=["id", f"{patient_ref}.reference"],
                        )
                    condition_df.dropna(
                        axis=0, inplace=True, how="any"
                    )  # Remove the rows with invalid values
                    condition_df.rename(
                        {
                            "id": "condition_id",
                            "patient.reference": "patient",
                            "subject.reference": "patient",
                        },
                        inplace=True,
                        axis=1,
                    )
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


if __name__ == "__main__":
    unittest.main()

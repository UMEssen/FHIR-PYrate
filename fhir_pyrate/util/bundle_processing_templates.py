from typing import Any, Callable, Dict, List, Tuple

from fhir_pyrate.util import FHIRObj


def flatten_data(bundle: FHIRObj, col_sep: str = "_") -> List[Dict]:
    """
    Preprocessing function that goes through the JSON bundle and returns lists of dictionaries
    for all possible attributes

    :param bundle: The bundle returned by the FHIR request
    :param col_sep: The separator to use to generate the column names for the DataFrame
    :return: A dictionary containing the parsed information
    """
    records = []
    for entry in bundle.entry or []:
        base_dict: Dict[str, Any] = {}
        recurse_resource(
            resource=entry.resource, base_dict=base_dict, field_name="", col_sep=col_sep
        )
        records.append(base_dict)
    return records


def recurse_resource(
    resource: Any, base_dict: Dict[str, Any], field_name: str, col_sep: str = "_"
) -> None:
    """
    Recursively go through the resource and store the values if they do not contain other
    sub-objects.

    :param resource: The resource to go through
    :param base_dict: The dictionary that will be filled
    :param field_name: The current name for the field
    :param col_sep: The separator between the attribute names
    """
    if isinstance(resource, FHIRObj):
        for attr, value in resource.__dict__.items():
            recurse_resource(
                resource=value,
                base_dict=base_dict,
                field_name=field_name + col_sep + attr,
                col_sep=col_sep,
            )
    elif isinstance(resource, List):
        for i, element in enumerate(resource):
            recurse_resource(
                resource=element,
                base_dict=base_dict,
                field_name=field_name + col_sep + str(i),
                col_sep=col_sep,
            )
    else:
        assert field_name[1:] not in base_dict, (
            f"The field {field_name[1:]} is already present in the dictionary with value "
            f"{base_dict[field_name[1:]]}"
        )
        # Always remove the first character, it is always a col_sep
        base_dict[field_name[1:]] = resource


def parse_fhir_path(
    bundle: FHIRObj, compiled_fhir_paths: List[Tuple[str, Callable]]
) -> List[Dict]:
    """
    Preprocessing function that goes through the JSON bundle and returns lists of dictionaries
    for all possible attributes, which have been specified using a list of compiled FHIRPath
    expressions (https://hl7.org/fhirpath/).

    :param bundle: The bundle returned by the FHIR request
    :param compiled_fhir_paths: A list of tuples of the form (column_name, compiled_fhir_paths),
    where column_name is the name of the future column, and fhir_paths is a compiled function
    that will be used to build that column and that was compiled from a FHIR path (
    https://hl7.org/fhirpath/). Please refer to the `bundles_to_dataframe`
    functions for notes on how to use the FHIR paths.
    :return: A dictionary containing the parsed information
    """
    records = []
    for entry in bundle.entry or []:
        resource = entry.resource
        base_dict: Dict[str, Any] = {}
        for name, compiled_path in compiled_fhir_paths:
            if name not in base_dict or base_dict[name] is None:
                base_dict[name] = compiled_path(resource=resource.to_dict())
            if len(base_dict[name]) == 0:
                base_dict[name] = None
            elif len(base_dict[name]) == 1:
                base_dict[name] = next(iter(base_dict[name]))
        records.append(base_dict)
    return records

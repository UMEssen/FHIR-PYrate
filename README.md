[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Supported Python version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Stable Version](https://img.shields.io/pypi/v/fhir-pyrate?label=stable)](https://pypi.org/project/fhir-pyrate/)
[![Pre-release Version](https://img.shields.io/github/v/release/UMEssen/fhir-pyrate?label=pre-release&include_prereleases&sort=semver)](https://pypi.org/project/fhir-pyrate/#history)
[![DOI](https://zenodo.org/badge/456893108.svg)](https://zenodo.org/badge/latestdoi/456893108)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/UMEssen/FHIR-PYrate">
    <img src="https://raw.githubusercontent.com/UMEssen/FHIR-PYrate/main/images/logo.svg" alt="Logo" width="440" height="338">
  </a>
</div>

This package is meant to provide a simple abstraction to query and structure FHIR resources as
pandas DataFrames. Want to use R instead? Try out [fhircrackr](https://github.com/POLAR-fhiR/fhircrackr)!

There are four main classes:
* [Ahoy](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/ahoy.py): Authenticate on the FHIR API
([Example 1](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/1-simple-json-to-df.ipynb),
[2](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/2-condition-to-imaging-study.ipynb)),
at the moment only BasicAuth and token authentication are supported.
* [Pirate](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/pirate.py): Extract and search for data via FHIR
  API
  ([Example 1](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/1-simple-json-to-df.ipynb),
[2](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/2-condition-to-imaging-study.ipynb),
[3](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/3-observation-for-condition.ipynb) &
[4](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/4-patients-for-diagnostic-report.ipynb)).
* [Miner](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/miner.py): Search for keywords or phrases
  within Diagnostic Report ([Example 4](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/4-patients-for-diagnostic-report.ipynb)).
* [DicomDownloader](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/dicom_downloader.py): Download complete studies or
  series ([Example 2](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/2-condition-to-imaging-study.ipynb)).

**DISCLAIMER**:
We have tried to add tests for some public FHIR servers. However, because of the quality and
quantity of resources we could not test as much as we have tested with the local FHIR server at
our institute. If there is anything in the code that only applies to our server, or you have
problems with the authentication (or anything else really), please just create an issue or
[email us](mailto:giulia.baldini@uk-essen.de).

<br />
<div align="center">
  <img src="https://raw.githubusercontent.com/UMEssen/FHIR-PYrate/main/images/resources.svg" alt="Resources" width="630" height="385">
</div>

<!-- TABLE OF CONTENTS -->
Table of Contents:

* [Install](https://github.com/UMEssen/FHIR-PYrate/#install)
   * [Either Pip](https://github.com/UMEssen/FHIR-PYrate/#either-pip)
   * [Or Within Poetry](https://github.com/UMEssen/FHIR-PYrate/#or-within-poetry)
* [Run Tests](https://github.com/UMEssen/FHIR-PYrate/#run-tests)
* [Explanations &amp; Examples](https://github.com/UMEssen/FHIR-PYrate/#explanations--examples)
   * [Ahoy](https://github.com/UMEssen/FHIR-PYrate/#ahoy)
   * [Pirate](https://github.com/UMEssen/FHIR-PYrate/#pirate)
      * [sail_through_search_space](https://github.com/UMEssen/FHIR-PYrate/#sail_through_search_space)
      * [trade_rows_for_bundles](https://github.com/UMEssen/FHIR-PYrate/#trade_rows_for_bundles)
      * [bundles_to_dataframe](https://github.com/UMEssen/FHIR-PYrate/#bundles_to_dataframe)
      * [***_dataframe](https://github.com/UMEssen/FHIR-PYrate/#_dataframe)
   * [Miner](https://github.com/UMEssen/FHIR-PYrate/#miner)
   * [DicomDownloader](https://github.com/UMEssen/FHIR-PYrate/#dicomdownloader)
* [Contributing](https://github.com/UMEssen/FHIR-PYrate/#contributing)
* [Authors and acknowledgment](https://github.com/UMEssen/FHIR-PYrate/#authors-and-acknowledgment)
* [License](https://github.com/UMEssen/FHIR-PYrate/#license)
* [Project status](https://github.com/UMEssen/FHIR-PYrate/#project-status)


## Install

### Either Pip
The package can be installed using PyPi
```bash
pip install fhir-pyrate
```
or using GitHub (always the newest version).
```bash
pip install git+https://github.com/UMEssen/FHIR-PYrate.git
```

These two commands only install the packages needed for **Pirate**. If you also want to use the **Miner** or the **DicomDownloader**, then you need to install them as extra dependencies with
```bash
pip install "fhir-pyrate[miner]" # only for miner
pip install "fhir-pyrate[downloader]" # only for downloader
pip install "fhir-pyrate[all]" # for both
```

### Or Within Poetry
We can also use poetry for this same purpose. Using PyPi we need to run the following commands.
```bash
poetry add fhir-pyrate
poetry install
```
Whereas to add it from GitHub, we have different options, because until recently
[poetry used to exclusively install from the master branch](https://github.com/python-poetry/poetry/issues/3366).

Poetry 1.2.0a2+:
```bash
poetry add git+https://github.com/UMEssen/FHIR-PYrate.git
poetry install
```
For the previous versions you need to add the following line to your `pyproject.toml` file:
```bash
fhir-pyrate = {git = "https://github.com/UMEssen/FHIR-PYrate.git", branch = "main"}
```
and then run
```bash
poetry lock
```

Also in poetry, the above only installs the packages for **Pirate**. If you also want to use the **Miner** or the **DicomDownloader**, then you need to install them as extra dependencies with
```bash
poetry add "fhir-pyrate[miner]" # only for miner
poetry add "fhir-pyrate[downloader]" # only for downloader
poetry add "fhir-pyrate[all]" # for both
```
or by adding the following to your `pyproject.toml` file:
```bash
fhir-pyrate = {git = "https://github.com/UMEssen/FHIR-PYrate.git", branch = "main", extras = ["all"]}
```

## Run Tests

When implementing new features, make sure that the existing ones have not been broken by using our
unit tests. First set the `FHIR_USER` and `FHIR_PASSWORD` environment variables with your
username and password for the FHIR server and then run the tests.

```bash
poetry run python -m unittest discover tests
```

If you implement a new feature, please add a small test for it in
[tests](https://github.com/UMEssen/FHIR-PYrate/blob/main/tests). You can
also use the tests as examples.

## Explanations & Examples

Please look at the [examples](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples) folder for complete examples.

### [Ahoy](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/ahoy.py)

The **Ahoy** class is used to authenticate and is needed for the **Pirate** and
**DicomDownloader** classes.

```python
from fhir_pyrate import Ahoy

# Authorize via password
auth = Ahoy(
  username="your_username",
  auth_method="password",
  auth_url="auth-url", # Your URL for authentication
  refresh_url="refresh-url", # Your URL to refresh the authentication token (if available)
)
```

We accept the following authentication methods:

* **token**: Pass your already generated token as a constructor argument.
* **password**: Enter your password via prompt.
* **env**: Use the `FHIR_USER` and `FHIR_PASSWORD` environment variables (mostly used for
  the unit tests). You can also change their names with the `change_environment_variable_name`
  function.
* **keyring**: To Be Implemented.

### [Pirate](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/pirate.py)

The **Pirate** can query any resource implemented within the FHIR API and is initialized as
follows:

```python
from fhir_pyrate import Pirate

auth = ...

# Init Pirate
search = Pirate(
    auth=auth,
    base_url="fhir-url", # e.g. "http://hapi.fhir.org/baseDstu2"
    print_request_url=False, # If set to true, you will see all requests
)
```

The Pirate functions do one of three things:
1. They run the query and collect the resources and store them in a generator of bundles.
   * `steal_bundles`: single process, no timespan to specify
   * `sail_through_search_space`: multiprocess, divide&conquer with many smaller timespans
   * `trade_rows_for_bundles`: multiprocess, takes DataFrame as input and runs one query per row
2. They take a generator of bundles and build a DataFrame.
   * `bundles_to_dataframe`: multiprocess, builds the DataFrame from the bundles.
3. They are wrapper that combine the functionalities of 1&2, or that set some particular parameters.
   * `steal_bundles_to_dataframe`: single process, executes `steal_bundles` and then runs `bundles_to_dataframe` on the result.
   * `sail_through_search_space_to_dataframe`: multiprocess, executes `sail_through_search_space` and then runs `bundles_to_dataframe` on the result.
   * `trade_rows_for_dataframe`: multiprocess, executes `trade_rows_for_bundles` and then runs `bundles_to_dataframe` on the result, it is also possible to add columns from the original DataFrame to the result

| Name                                    | Type | Multiprocessing | DF Input? |        Output        |
|:----------------------------------------|:----:|:---------------:|:---------:|:--------------------:|
| steal_bundles                           |  1   |       No        |    No     | Generator of FHIRObj |
| sail_through_search_space               |  1   |       Yes       |    No     | Generator of FHIRObj |
| trade_rows_for_bundles                  |  1   |       Yes       |    Yes    | Generator of FHIRObj |
| bundles_to_dataframe                    |  2   |       Yes       |     /     |      DataFrame       |
| steal_bundles_to_dataframe              |  3   |       No        |    No     |      DataFrame       |
| sail_through_search_space_to_dataframe  |  3   |       Yes       |    No     |      DataFrame       |
| trade_rows_for_dataframe                |  3   |       Yes       |    Yes    |      DataFrame       |


**BETA FEATURE**: It is also possible to cache the bundles using the `bundle_caching` parameter,
which specifies a caching folder. This has not yet been tested extensively and does not have any
cache invalidation mechanism.


A toy request for ImagingStudy:

```python
search = ...

# Make the FHIR call
bundles = search.sail_through_search_space_to_dataframe(
    resource_type="ImagingStudy",
    date_init="2021-04-01",
    time_attribute_name="started",
    request_params={
      "modality": "CT",
      "_count": 5000,
    }
)
```

The argument `request_params` is a dictionary that takes a string as key (the FHIR identifier) and anything as value.
If the value is a list or tuple, then all values will be used to build the request to the FHIR API.

`sail_through_search_space_to_dataframe` is a wrapper function that directly converts the result of
`sail_through_search_space` into a DataFrame.

#### [`sail_through_search_space`](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/pirate.py)

The `sail_through_search_space` function uses the multiprocessing module to speed up some queries.
The multiprocessing is done as follows:
The time frame is divided into multiple time spans (as many as there are processes) and each smaller
time frame is investigated simultaneously. This is why it is necessary to give a `date_init`
and `date_end` param to the `sail_through_search_space` function.

**Note** that if the `date_init` or `date_end` parameters are given as strings, they will be converted
to `datetime.datetime` objects, so any non specified parameters (month, day or time) will be assumed
according to the `datetime` workflow, and then converted to string according to the `time_format`
specified in the **Pirate** constructor.

A problematic aspect of the resources is that the date in which the resource was acquired is defined
using different attributes. Also, some resources use a fixed date, other use a time period.
You can specify the date attribute that you want to use with `time_attribute_name`.

The resources where the date is based on a period (such as `Encounter` or `Procedure`) may cause
duplicates in the multiprocessing because one entry may belong to multiple time spans that are
generated. You can drop the ID duplicates once you have built a DataFrame with your data.

#### [`trade_rows_for_bundles`](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/pirate.py)

In case we already have an Excel sheet or CSV file with `fhir_patient_id`s or any other
identifier), and we want to request resources based on those
identifiers we can use the function `trade_rows_for_bundles`:

```python
search = ...
# DataFrame containing FHIR patient IDs
patient_df = ...

# Collect all imaging studies defined within df_reports
dr_bundles = search.trade_rows_for_bundles(
  patient_df,
  resource_type="DiagnosticReport",
  request_params={"_count": "100", "status": "final"},
  df_constraints={"subject": "fhir_patient_id"},
)
```

We only have to define the `resource_type` and the constraints that we want to enforce from the
DataFrame in `df_constraints`. This dictionary should contain pairs of (`fhir_identifier`,
`identifier_column`) where `fhir_identifier` is the API search parameter and `identifier_column`
is the column where the values that we want to search for are stored.
Additionally, a system can be used to better identify the constraints of the DataFrame.
For example, let us assume that we have a column of the DataFrame (called `loinc_code` that
contains a bunch of different LOINC codes. Our `df_constraints` could look as follows:
```
df_constraints={"code": ("http://loinc.org", "loinc_code")}
```

This function also uses multiprocessing, but differently from before, it will process the rows
of the DataFrame in parallel.

#### [`bundles_to_dataframe`](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/pirate.py)

The two functions described above return a generator of `FHIRObj` bundles which can then be
converted to a `DataFrame` using this function.

The `bundles_to_dataframe` has three options on how to handle and extract the relevant information
from the bundles:
1. Extract everything, in this case you can use the
[`flatten_data`](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/util/bundle_processing_templates.py)
function, which is already the default for `process_function`, so you do not actually need to
specify anything.
```python
# Create bundles with Pirate
search = ...
bundles = ...
# Convert the returned bundles to a dataframe
df = search.bundles_to_dataframe(
    bundles=bundles,
)
```
2. Use a processing function where you define exactly which attributes are needed by iterating
   through the entries and selecting the elements. The values that will be added to the
   dictionary represent the columns of the DataFrame. For an example of when it might make sense
   to do this, check [Example 3](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/3-patients-for-condition.ipynb).
```python
from typing import List, Dict
from fhir_pyrate.util.fhirobj import FHIRObj
# Create bundles with Pirate
search = ...
bundles = ...
def get_diagnostic_text(bundle: FHIRObj) -> List[Dict]:
    records = []
    for entry in bundle.entry or []:
        resource = entry.resource
        records.append(
            {
                "fhir_diagnostic_report_id": resource.id,
                "report_status": resource.text.status,
                "report_text": resource.text.div,
            }
        )
    return records
# Convert the returned bundles to a dataframe
df = search.bundles_to_dataframe(
    bundles=bundles,
    process_function=get_diagnostic_text,
)
```
3. Extract only part of the information using the `fhir_paths` argument. Here you can put a list
   of string that follow the [FHIRPath](https://hl7.org/fhirpath/) standard. For this purpose, we
   use the [fhirpath-py](https://github.com/beda-software/fhirpath-py) package, which uses the
   [antr4](https://github.com/antlr/antlr4) parser. Additionally, you can use tuples like `(key,
   fhir_path)`, where `key` will be the name of the column the information derived from that
   FHIRPath will be stored.
```python
# Create bundles with Pirate
search = ...
bundles = ...
# Convert the returned bundles to a dataframe
df = search.bundles_to_dataframe(
    bundles=bundles,
    fhir_paths=["id", ("code", "code.coding"), ("identifier", "identifier[0].code")],
)
```
**NOTE 1 on FHIR paths**: The standard also allows some primitive math operations such as modulus
(`mod`) or integer division (`div`), and this may be problematic if there are fields of the
resource that use these terms as attributes.
It is actually the case in many generated [public FHIR resources](https://hapi.fhir.org/baseDstu2/DiagnosticReport/133015).
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
    fhir_paths=[("patient", "Patient.id"), ("patient", "subject.reference")],
    # WRONG EXAMPLE
    # In this case, only the first code will be stored
    fhir_paths=[("code", "code.coding[0].code"), ("code", "code.coding[1].code")],
    # CORRECT EXAMPLE
    # Whenever we are working with codes, it is usually better to use the `where` argument and
    # to store the values using a meaningful name
    fhir_paths=[
        ("code_abc", "code.coding.where(system = 'ABC').code"),
        ("code_def", "code.coding.where(system = 'DEF').code"),
    ],
    num_pages=1,
)
```

#### [`***_dataframe`](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/pirate.py)
The `steal_bundles_to_dataframe`, `sail_through_search_space_to_dataframe` and `trade_rows_for_dataframe`
are facade functions which retrieve the bundles and then run `bundles_to_dataframe`.

In `trade_rows_for_dataframe` you can also specify the `with_ref` parameter to also add the
parameters specified in `df_constraints` as columns of the final DataFrame.
You can find an example in [Example 3](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/3-patients-for-condition.ipynb).
Additionally, you can specify the `with_columns` parameter, which can add any columns from the original
DataFrame. The columns can be either specified as a list of columns `[col1, col2, ...]` or as a
list of tuples `[(new_name_for_col1, col1), (new_name_for_col2, col2), ...]`

### [Miner](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/miner.py)

<br />
<div align="center">
  <img src="https://raw.githubusercontent.com/UMEssen/FHIR-PYrate/main/images/miner.svg" alt="Logo" width="718" height="230">
</div>
<br />

The **Miner** takes a DataFrame and searches it for a particular regular expression
with the help of [SpaCy](https://spacy.io/).
It is also possible to add a regular expression for the text that should be excluded.
Please use a RegEx checker (e.g. [https://regex101.com/](https://regex101.com/)) to build your
regular expressions.

```python
from fhir_pyrate import Miner

df_diagnostic_reports = ...  # Get a DataFrame
# Search for text where the word "Tumor" is present
miner = Miner(
    target_regex="Tumor*",
    decode_text=...# Here you can write a function that processes each single text (e.g. stripping, decoding)
)
df_filtered = miner.nlp_on_dataframe(
  df_diagnostic_reports,
  text_column_name="report_text",
  new_column_name="text_found"
)
```

### [DicomDownloader](https://github.com/UMEssen/FHIR-PYrate/blob/main/fhir_pyrate/dicom_downloader.py)

At our institute we have a DicomWebAdapter app that can be used to download studies and series
from the PACS system of our hospital. The DicomDownloader uses the
[DicomWebClient](https://dicomweb-client.readthedocs.io/en/latest/usage.html) with a specific
internal URL for each PACS to connect and download the images.
We could not find a public system that was offering anything similar, so this class has only
been tested on our internal FHIR server.
In case you have questions or you would like some particular features to be able to use this at
your institute, please do not hesitate and contact us, or write a pull request!

The **DicomDownloader** downloads a complete Study (StudyInstanceUID) or a specific series (
StudyInstanceUID + SeriesInstanceUID).

The relevant data can be downloaded either es DICOM (`.dcm`) or NIfTI (`.nii.gz`).
In the NIfTI case there will be an  additional `.dcm` file to store some metadata.

Using the function `download_data_from_dataframe` it is possible to download studies and series
directly from the data of a given dataframe. The column that contain the study/series
information can be specified. To have an example of how the DataFrame should look like, please
refer to [Example 2](https://github.com/UMEssen/FHIR-PYrate/blob/main/examples/2-condition-to-imaging-study.ipynb).
A DataFrame will be returned which specifies the successfully downloaded Study/Series ID, the
deidentified IDs and the download folder name. Additionally, a DataFrame containing the failed
studies will also be returned, together with the kind of error and the traceback.

```python
from fhir_pyrate import DicomDownloader

auth = ...
# Initialize the Study Downloader
# Decide to download the data as NIfTis, set it to "dicom" for DICOMs
downloader = DicomDownloader(
  auth=auth,
  output_format="nifti",
  dicom_web_url=DICOM_WEB_URL, # Specify a URL of your DICOM Web Adapter
)

# Get some studies
df_studies = ...
# Download the series
successful_df, error_df = downloader.download_data_from_dataframe(
  df_studies,
  output_dir="out",
  study_uid_col="study_instance_uid",
  series_uid_col="series_instance_uid",
  download_full_study=False, # If we download the entire study, series_instance_uid will not be used
)
```

Additionally, it is also possible to use the `download_data` function to download a single study or
series given as parameter.
In this case, the mapping information will be returned as a list of dictionaries that can be used
to build a mapping file.

```python
# Download only one series and get some download information
download_info = downloader.download_data(
  study_uid="1.2.826.0.1.3680043.8.498.24222694654806877939684038520520717689",
  series_uid="1.2.826.0.1.3680043.8.498.33463995182843850024561469634734635961",
  output_dir="out",
  save_metadata=True,
)
# Download only one full study
download_info_study = downloader.download_data(
  study_uid="1.2.826.0.1.3680043.8.498.24222694654806877939684038520520717689",
  series_uid=None,
  output_dir="out",
  save_metadata=True,
)
```

## Contributing
<!-- Thank you https://github.com/othneildrew/Best-README-Template -->
Contributions are what make the open source community such an amazing place to learn, inspire, and create.
Any contributions you make are greatly appreciated.
If you have a suggestion that would make this better, please fork the repo and create a pull
request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## Authors and acknowledgment

This package was developed by the [SHIP-AI group at the Institute for Artificial Intelligence in Medicine](https://ship-ai.ikim.nrw/).

- [goku1110](https://github.com/goku1110): initial idea, development, logo & figures
- [giuliabaldini](https://github.com/giuliabaldini): development, tests, new features

We would like to thank [razorx89](https://github.com/razorx89), [butterpear](https://github.com/butterpear), [vkyprmr](https://github.com/vkyprmr), [Wizzzard93](https://github.com/Wizzzard93), [karzideh](https://github.com/karzideh) and [luckfamousa](https://github.com/luckfamousa) for their input, time and effort.

## License
This project is licenced under the [MIT Licence](LICENSE).

## Project status
The project is in active development.

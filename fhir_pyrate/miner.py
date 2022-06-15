import logging
import multiprocessing
import re
import subprocess
import traceback
from functools import partial
from typing import Callable, List, Optional, Union

import pandas as pd
import spacy
from spacy.tokens import Span
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Miner:
    """
    Searches the reports for a particular regex.

    :param target_regex: The target regex to be used to filter the text
    :param negation_regex: The regex that should be added to negate the target regex
    :param decode_text: Function to use to decode the document's text
    :param nlp_lib: Name of the NLP library that should be downloaded by SpaCy
    :param num_processes: The number of processes to run the NLP in multiprocessing
    """

    def __init__(
        self,
        target_regex: str,
        negation_regex: str = None,
        regex_flags: Union[int, re.RegexFlag] = None,
        decode_text: Callable = None,
        nlp_lib: str = "de_core_news_sm",
        num_processes: int = 1,
    ) -> None:
        self.target_regex = target_regex
        self.negation_regex = negation_regex
        self.regex_flags = regex_flags if regex_flags is not None else 0
        self.decode_text = decode_text
        try:
            self.nlp = spacy.load(nlp_lib)
        except IOError:
            # NOTE: Run python -m spacy download {nlp_lib} in your docker file
            # if you are using docker
            logger.warning(
                "If you are trying to install the spacy library within docker, "
                "this will probably not work, because it needs access to your home "
                "directory. Please run python -m spacy download {nlp_lib} in your "
                "docker file.",
            )
            subprocess.run(
                f"python3 -m spacy download {nlp_lib}".split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
                check=True,
                universal_newlines=True,
            )
            self.nlp = spacy.load(nlp_lib)
            logger.debug(traceback.format_exc())

        self.num_processes = num_processes

    @staticmethod
    def _remove_header(sentences: List[Span], main_document_keyword: str) -> List[Span]:
        """
        Removes all sentences that come before a sentence that contains the `main_document_keyword`.
        This is useful when a document has a header, and we know what the first viable word of a
        document is, or we know that we are interested in some particular part of the
        document that comes after a certain keyword.

        :param sentences: List of sentences within the report
        :param main_document_keyword: The keyword after which the interesting part of the document
        begins
        :return: List of sentences that are after the keyword (inclusive)
        """
        id_filter = 0
        for i, x in enumerate(sentences):
            if main_document_keyword in x.text:
                id_filter = i
                break
        return sentences[id_filter:]

    def _check_diagnostic_report(
        self,
        report_text: str,
        main_document_keyword: str = "",
    ) -> Optional[List[Span]]:
        """
        Checks whether a report contains the relevant RegEx and does not contain the negation
        RegEx (if specified).

        :param report_text: The text to be searched
        :param main_document_keyword: The keyword after which the interesting part of the document
        begins
        :return: Returns a list of SpaCy Spans that match the RegEx, but do not match the
        negation regex, or None if no sentences were found
        """
        if self.decode_text is not None:
            report_text = self.decode_text(report_text)
        contains_target = re.search(self.target_regex, report_text, self.regex_flags)
        relevant_sentences = []
        if contains_target:
            sentences = [i for i in self.nlp(report_text).sents]
            if main_document_keyword is not None:
                sentences = self._remove_header(sentences, main_document_keyword)

            relevant_sentences = [
                x
                for x in sentences
                if re.search(self.target_regex, x.text, self.regex_flags) is not None
            ]
            if self.negation_regex is not None:
                negation_sentences = [
                    re.search(self.negation_regex, x.text, self.regex_flags) is not None
                    for x in relevant_sentences
                ]
                relevant_sentences = [
                    x
                    for i, x in enumerate(relevant_sentences)
                    if not negation_sentences[i]
                ]
        return relevant_sentences if len(relevant_sentences) > 0 else None

    def nlp_on_dataframe(
        self,
        df: pd.DataFrame,
        text_column_name: str,
        new_column_name: str = "text_found",
        main_document_keyword: str = None,
    ) -> pd.DataFrame:
        """
        Searches the strings contained in `text_column_name` for the selected RegEx, and adds two
        columns to the DataFrame with the output of the NLP search. The negation RegEx can be
        used to exclude sentences. Additionally, it is possible to define a `main_document_keyword`,
        which is a string that can be used to filter out the header of the document.

        :param df: Dataframe containing all strings that should be searched
        :param text_column_name: The column that should be searched
        :param new_column_name: The name of the new column that will be added to indicate whether the
        RegEx was found and to display the sentences.
        :param main_document_keyword: String that can be used to filter out some sentences that do not
        contain interesting information
        :return: The input DataFrame with two new columns: The `new_column_name` tells us
        whether the desired text was found or not, the `new_column_name`_sentences column returns a
        List of sentences (in the form of already processed SpaCy Spans).
        """
        func = partial(
            self._check_diagnostic_report,
            main_document_keyword=main_document_keyword,
        )
        texts = [row for row in df[text_column_name].values]
        tqdm_text = f"Searching for Sentences with {self.target_regex}"
        if self.negation_regex is not None:
            tqdm_text += f" and without {self.negation_regex}"
        if self.num_processes > 1:
            pool = multiprocessing.Pool(self.num_processes)
            results = [
                result
                for result in tqdm(
                    pool.imap(func, texts),
                    total=len(df),
                    desc=tqdm_text,
                )
            ]
            pool.close()
            pool.join()
        else:
            results = [
                result
                for result in tqdm(
                    [func(text) for text in texts],
                    total=len(df),
                    desc=tqdm_text,
                )
            ]

        df[new_column_name + "_sentences"] = results
        df[new_column_name] = ~df[new_column_name + "_sentences"].isna()
        return df

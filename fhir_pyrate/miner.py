import logging
import multiprocessing
import re
import subprocess
import traceback
from functools import partial
from typing import Callable, List

import pandas as pd
import spacy
from spacy.tokens import Span
from tqdm import tqdm


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
        negation_regex: str = r"(kein|keine|ohne)[^.]*",
        decode_text: Callable = None,
        nlp_lib: str = "de_core_news_sm",
        num_processes: int = 1,
    ) -> None:
        self.target_regex = target_regex
        self.negation_regex = negation_regex
        self.decode_text = decode_text
        try:
            self.nlp = spacy.load(nlp_lib)
        except IOError:
            # NOTE: Run python -m spacy download {nlp_lib} in your docker file
            # if you are using docker
            logging.warning(
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
            logging.debug(traceback.format_exc())

        self.num_processes = num_processes

    @staticmethod
    def _filter_report_header(sentences: List[Span], filter_text: str) -> List[Span]:
        """
        Filters a report according to some target text.

        :param sentences: List of sentences within the report
        :param filter_text: Target text which will be filtered
        :return: Filtered sentences list
        """
        start_idx = [idx for idx, x in enumerate(sentences) if filter_text in x.text]
        return sentences[start_idx[0] : -1] if len(start_idx) > 0 else sentences

    def _check_diagnostic_report(
        self,
        report_text: str,
        filter_text: str = "",
    ) -> bool:
        """
        Checks whether a report contains the relevant keyword.

        :param report_text: The text to be searched
        :param filter_text: String that can be used to filter out some sentences that do not
        contain interesting information
        :return: Returns bool which will be true if the report contains the relevant keywords
        """
        is_target = False
        if self.decode_text is not None:
            report_text = self.decode_text(report_text)
        report_text = report_text.lower()
        contains_target = re.search(self.target_regex, report_text, re.I | re.M)

        if contains_target:
            sentences = [i for i in self.nlp(report_text).sents]
            sentences = self._filter_report_header(sentences, filter_text=filter_text)

            relevant_sentences = [
                x
                for x in sentences
                if re.search(self.target_regex, x.text, re.I | re.M) is not None
            ]
            negation_sentences = [
                re.search(self.negation_regex, x.text, re.I | re.M) is not None
                for x in relevant_sentences
            ]
            is_target = not any(negation_sentences)

        return is_target

    def nlp_on_dataframe(
        self,
        df: pd.DataFrame,
        text_column_name: str,
        new_column_name: str = "text_found",
        filter_text: str = "",
    ) -> pd.DataFrame:
        """
        Add a new column to our DataFrame with the output of the NLP search.

        :param df: Dataframe containing all reports
        :param text_column_name: The column that should be searched
        :param new_column_name: The name of the new column that will be added
        :param filter_text: String that can be used to filter out some sentences that do not
        contain interesting information
        :return: The input DataFrame with a new column telling us whether the desired text was
        found or not.
        """
        if self.num_processes > 1:
            pool = multiprocessing.Pool(self.num_processes)
            df[new_column_name] = [
                result
                for result in tqdm(
                    pool.imap(
                        partial(
                            self._check_diagnostic_report,
                            filter_text=filter_text,
                        ),
                        [row[text_column_name] for _, row in df.iterrows()],
                    ),
                    total=len(df),
                    desc=f"Searching for {self.target_regex}",
                )
            ]
            pool.close()
            pool.join()
        else:
            df[new_column_name] = [
                result
                for result in tqdm(
                    [
                        self._check_diagnostic_report(
                            row[text_column_name], filter_text=filter_text
                        )
                        for _, row in df.iterrows()
                    ],
                    total=len(df),
                    desc=f"Searching for {self.target_regex}",
                )
            ]

        return df

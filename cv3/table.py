"""Table class: a wrapper over pandas.DataFrame
"""

from pandas import DataFrame

import pandas as pd


class Table:
    """Table class: a wrapper over pandas.DataFrame
    """

    def __init__(self, source: DataFrame):
        """Creates table class

        Args:
            source: pandas DataFrame
        """

        self.dataframe = source

    @property
    def columns(self) -> tuple:

        return tuple(self.dataframe.columns)

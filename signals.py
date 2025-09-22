import pandas as pd

class EntryRule:
    def check(self, ts, price, df):
        """
        Return (True, params_dict) if entry triggered at this bar.
        """
        raise NotImplementedError


class CurrencyPair:
    def __init__(self, symbol, pip_size=0.01, contract_size=100_000, quote_currency="USD"):
        self.symbol = symbol
        self.pip_size = pip_size
        self.contract_size = contract_size
        self.quote_currency = quote_currency

    def pip_value_per_lot(self, price):
        return (self.contract_size * self.pip_size) / price

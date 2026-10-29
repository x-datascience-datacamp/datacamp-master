import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve


URL = (
    'https://raw.githubusercontent.com/scikit-learn/examples-data/'
    'master/financial-data/{}.csv'
)
SYMBOLS = {
    'TOT': 'Total',
    'XOM': 'Exxon',
    'CVX': 'Chevron',
    'COP': 'ConocoPhillips',
    'VLO': 'Valero Energy',
    'MSFT': 'Microsoft',
    'IBM': 'IBM',
    'TWX': 'Time Warner',
    'CMCSA': 'Comcast',
    'CVC': 'Cablevision',
    'YHOO': 'Yahoo',
    'DELL': 'Dell',
    'HPQ': 'HP',
    'AMZN': 'Amazon',
    'TM': 'Toyota',
    'CAJ': 'Canon',
    'SNE': 'Sony',
    'F': 'Ford',
    'HMC': 'Honda',
    'NAV': 'Navistar',
    'NOC': 'Northrop Grumman',
    'BA': 'Boeing',
    'KO': 'Coca Cola',
    'MMM': '3M',
    'MCD': 'McDonald\'s',
    'PEP': 'Pepsi',
    'K': 'Kellogg',
    'UN': 'Unilever',
    'MAR': 'Marriott',
    'PG': 'Procter Gamble',
    'CL': 'Colgate-Palmolive',
    'GE': 'General Electrics',
    'WFC': 'Wells Fargo',
    'JPM': 'JPMorgan Chase',
    'AIG': 'AIG',
    'AXP': 'American express',
    'BAC': 'Bank of America',
    'GS': 'Goldman Sachs',
    'AAPL': 'Apple',
    'SAP': 'SAP',
    'CSCO': 'Cisco',
    'TXN': 'Texas Instruments',
    'XRX': 'Xerox',
    'WMT': 'Wal-Mart',
    'HD': 'Home Depot',
    'GSK': 'GlaxoSmithKline',
    'PFE': 'Pfizer',
    'SNY': 'Sanofi-Aventis',
    'NVS': 'Novartis',
    'KMB': 'Kimberly-Clark',
    'R': 'Ryder',
    'GD': 'General Dynamics',
    'RTN': 'Raytheon',
    'CVS': 'CVS',
    'CAT': 'Caterpillar',
    'DD': 'DuPont de Nemours'}


def fetch_quote_data(symbols='all', data_dir='data'):
    """Fetch quote_data from american compagnies as a dataframe.

    Parameters
    ----------
    symbols : list or str
        List of quotes to fetch. Default to `all` available.
    data_dir : str or Path
        Path to store the data.
    """

    if symbols == 'all':
        symbols = list(SYMBOLS.keys())

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    quotes = pd.DataFrame()
    N = len(symbols)
    for i, symbol in enumerate(symbols):
        print(
            f"Loading quote for {symbol:4s} ({i} / {N})\r",
            end='', flush=True
        )
        name = SYMBOLS[symbol]

        filename = data_dir / f"{symbol}.csv"
        if not filename.exists():
            urlretrieve(URL.format(symbol), filename)
        this_quote = pd.read_csv(filename)
        quotes[name] = this_quote['open']
    print(f"Loaded the quotes for {symbols}")
    quotes.index = pd.to_datetime(this_quote['date'])
    return quotes

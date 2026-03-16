import pandas as pd
import os

def load_exchange_rates(csv_path='./app/static/Webstat_Export_fr.csv'):
    """
    Load exchange rates from CSV file.
    Assumes columns: 'Currency' and 'Rate' (rate to EUR).
    Returns a dict like {'USD': 1.05, 'EUR': 1.0, 'GBP': 0.85}
    """
    try:
        rates_df = pd.read_csv(csv_path)
        if rates_df.empty:
            raise ValueError("Exchange rates CSV is empty.")
        # Assume columns are 'Currency' and 'Rate'
        exchange_rates = dict(zip(rates_df['Currency'], rates_df['Rate']))
        return exchange_rates
    except FileNotFoundError:
        raise FileNotFoundError(f"Exchange rates file not found at {csv_path}")
    except KeyError as e:
        raise KeyError(f"Expected column not found in CSV: {e}")

# Currency symbols for display
CURRENCY_SYMBOLS = {
    'USD': '$',
    'EUR': '€',
    'GBP': '£'
}

def convert_price(price_usd, target_currency, exchange_rates):
    """
    Convert price from USD to target currency using exchange rates.
    """
    rate = exchange_rates.get(target_currency, 1.0)  # Default to 1 if not found
    return price_usd * rate

def get_currency_symbol(currency):
    """
    Get the symbol for the currency.
    """
    return CURRENCY_SYMBOLS.get(currency, currency)
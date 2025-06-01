# sp500_data_downloader.py

import yfinance as yf
import pandas as pd
from datetime import datetime

def download_sp500_data(start_date="2000-01-01", end_date=None, filename="sp500_data.csv"):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"Downloading S&P 500 data from {start_date} to {end_date}...")

    # Download using yfinance
    data = yf.download("^GSPC", start=start_date, end=end_date, progress=True)

    # Save to CSV
    data.to_csv(filename)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    download_sp500_data()

import pandas as pd

def clean_messy_sp500(file_path="sp500_data.csv", output_path="sp500_data_cleaned.csv"):
    # Load with MultiIndex columns
    df = pd.read_csv(file_path, header=[0, 1])

    # Flatten column names: ('Close', '^GSPC') to 'Close'
    df.columns = [col[0] if col[0] != 'Price' else 'Date' for col in df.columns]

    # Drop first row which is all NaNs 
    df = df[pd.to_numeric(df["Close"], errors='coerce').notna()]

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])
    df.reset_index(drop=True, inplace=True)

    # Reorder
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[cols]

    # Save cleaned CSV
    df.to_csv(output_path, index=False)
    print(f"Cleaned file saved to {output_path}")

if __name__ == "__main__":
    clean_messy_sp500()

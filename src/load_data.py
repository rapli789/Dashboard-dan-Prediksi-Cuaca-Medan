import pandas as pd


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Membaca file dataset cuaca Medan dari Excel."""
    df = pd.read_excel(file_path)
    return df

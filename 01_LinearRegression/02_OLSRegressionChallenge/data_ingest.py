import pandas as pd

class IngestData:
    def __init__(self) -> None:
        self.file_path = None

    def get_data(self, file_path: str) -> pd.DataFrame:
        self.file_path = file_path
        df = pd.read_csv(self.file_path)
        return df


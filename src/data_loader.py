"""
Data-loading utilities for AdMIRe 2.0 TSV files.

Each TSV row contains a multilingual compound expression, a sentence using
that expression, five image file names, and five English image captions.
The loader yields clean dictionaries ready for downstream consumption.
"""

from pathlib import Path
from typing import Dict, Generator, List, Any

import pandas as pd

class TSVLoader:

    _CAPTION_COLS: List[str] = [
        "image1_caption",
        "image2_caption",
        "image3_caption",
        "image4_caption",
        "image5_caption",
    ]

    _IMAGE_NAME_COLS: List[str] = [
        "image1_name",
        "image2_name",
        "image3_name",
        "image4_name",
        "image5_name",
    ]

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self._df: pd.DataFrame = pd.read_csv(self.filepath, sep="\t")

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame for inspection."""
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """Yield one dictionary per TSV row.

        Yields
        ------
        dict
            Keys: ``compound``, ``sentence``, ``captions`` (list of 5 str),
            ``image_names`` (list of 5 str).
        """
        for _, row in self._df.iterrows():
            captions: List[str] = [
                str(row[col]) for col in self._CAPTION_COLS
            ]
            image_names: List[str] = [
                str(row[col]) for col in self._IMAGE_NAME_COLS
            ]
            yield {
                "compound": str(row["compound"]),
                "sentence": str(row["sentence"]),
                "captions": captions,
                "image_names": image_names,
            }

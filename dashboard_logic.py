"""Small, testable transformations used by the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd


def filter_dataset(
    frame: pd.DataFrame,
    districts: list[str] | None = None,
    products: list[str] | None = None,
    year_range: tuple[int, int] | None = None,
    search: str = "",
) -> pd.DataFrame:
    """Return matching rows without modifying the uploaded source data.

    Empty multi-selects mean "all", which makes the first dashboard visit useful
    without asking people to select every category before seeing their data.
    """
    result = frame.copy()

    if districts and "Kecamatan" in result.columns:
        result = result[result["Kecamatan"].isin(districts)]
    if products and "Jenis Pupuk" in result.columns:
        result = result[result["Jenis Pupuk"].isin(products)]
    if year_range is not None and "Tahun" in result.columns:
        start, end = year_range
        result = result[result["Tahun"].between(start, end)]

    query = search.strip().casefold()
    if query and not result.empty:
        searchable = result.astype("string").fillna("")
        matches = searchable.apply(
            lambda column: column.str.casefold().str.contains(query, regex=False)
        ).any(axis=1)
        result = result[matches]

    return result.reset_index(drop=True)

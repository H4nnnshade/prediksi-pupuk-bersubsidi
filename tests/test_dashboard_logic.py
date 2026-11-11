import pandas as pd

from dashboard_logic import filter_dataset


def sample_frame():
    return pd.DataFrame(
        {
            "Tahun": [2023, 2023, 2024, 2024],
            "Kecamatan": ["Gerung", "Sekotong", "Gerung", "Labuapi"],
            "Jenis Pupuk": ["Urea", "NPK", "NPK", "Urea"],
            "RDKK (ton)": [100, 80, 120, 60],
            "Alokasi (ton)": [90, 75, 110, 55],
            "Realisasi (ton)": [85, 70, 95, 50],
        }
    )


def test_filter_dataset_applies_district_product_and_year_range_together():
    result = filter_dataset(
        sample_frame(),
        districts=["Gerung"],
        products=["NPK", "Urea"],
        year_range=(2024, 2024),
    )

    assert result[["Tahun", "Kecamatan", "Jenis Pupuk"]].to_dict("records") == [
        {"Tahun": 2024, "Kecamatan": "Gerung", "Jenis Pupuk": "NPK"}
    ]


def test_filtering_missing_category_columns_skips_unavailable_filters():
    frame = pd.DataFrame({"Tahun": [2023, 2024], "Catatan": ["A", "B"]})

    result = filter_dataset(
        frame, districts=["Gerung"], products=["Urea"], year_range=(2024, 2024)
    )

    assert result.to_dict("records") == [{"Tahun": 2024, "Catatan": "B"}]


def test_empty_filter_selections_mean_all_values():
    source = sample_frame()

    result = filter_dataset(source, districts=[], products=[], year_range=None)

    pd.testing.assert_frame_equal(result.reset_index(drop=True), source)


def test_search_is_case_insensitive_and_matches_any_visible_text_column():
    result = filter_dataset(sample_frame(), search="seko")

    assert result["Kecamatan"].tolist() == ["Sekotong"]


def test_filter_result_does_not_mutate_source_or_keep_sparse_index():
    source = sample_frame()

    result = filter_dataset(source, districts=["Labuapi"])

    assert result.index.tolist() == [0]
    assert len(source) == 4
    assert source["Kecamatan"].tolist()[-1] == "Labuapi"


def test_filter_dataset_handles_empty_input():
    result = filter_dataset(sample_frame().iloc[:0], search="anything")

    assert result.empty
    assert list(result.columns) == list(sample_frame().columns)

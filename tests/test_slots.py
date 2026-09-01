from pipeline.slots import extract_region_scope, extract_time_window


def test_extract_time_window_last_n_months():
    tw = extract_time_window("What is current demand for the last 2 months?")
    assert tw.kind == "relative_months"
    assert tw.n == 2
    assert tw.direction == "past"

    tw = extract_time_window("Show demand for the last 6 months.")
    assert tw.n == 6

    tw = extract_time_window("What is demand for the last two months?")
    assert tw.n == 2


def test_extract_time_window_last_n_quarters_and_years():
    tw = extract_time_window("Show semiconductor demand for the last 3 quarters.")
    assert tw.kind == "relative_quarters"
    assert tw.n == 3

    tw = extract_time_window("What is demand for the last 2 years?")
    assert tw.kind == "relative_years"
    assert tw.n == 2


def test_extract_time_window_past_year_with_no_explicit_digit():
    tw = extract_time_window("What is semiconductor demand for the past year?")
    assert tw.kind == "relative_years"
    assert tw.n is None
    assert tw.direction == "past"


def test_extract_time_window_bare_last_months_has_no_explicit_n():
    tw = extract_time_window("What is current demand for the last months?")
    assert tw.kind == "relative_months"
    assert tw.n is None


def test_extract_time_window_future_phrasing():
    tw = extract_time_window("What is expected demand for the first upcoming quarter?")
    assert tw.kind == "relative_quarters"
    assert tw.direction == "future"

    tw = extract_time_window("Show demand for the next 2 quarters.")
    assert tw.kind == "relative_quarters"
    assert tw.n == 2
    assert tw.direction == "future"


def test_extract_time_window_absent_when_no_relative_phrase():
    tw = extract_time_window("Show current demand by region.")
    assert tw.kind is None
    assert tw.n is None
    assert tw.is_present is False


def test_extract_region_scope_combined_regions():
    assert extract_region_scope("What is the combined demand for Europe and America?") == {
        "europe",
        "americas",
    }
    assert extract_region_scope("Show demand for the Americas.") == {"americas"}
    assert extract_region_scope("Show demand by China and Japan.") == {"china", "japan"}


def test_extract_region_scope_absent_when_no_region_mentioned():
    assert extract_region_scope("What is total current demand?") == set()


def test_extract_region_scope_oem_vs_tier1_is_not_a_region():
    # OEM vs Tier1 is a survey-origin scope, not a region; extract_region_scope
    # should not misclassify it.
    assert extract_region_scope("Compare OEM vs Tier1 current demand.") == set()

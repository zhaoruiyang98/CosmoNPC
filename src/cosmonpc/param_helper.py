import numpy as np


def validate_tracer(tracer_type, correlation_mode):
    assert correlation_mode in [
        "auto",
        "cross",
    ], "correlation_mode must be either 'auto' or 'cross'"
    assert tracer_type in [
        "aaa",
        "aab",
        "abb",
        "abc",
    ], "tracer_type must be one of 'aaa', 'aab', 'abb', or 'abc'"
    if correlation_mode == "auto":
        assert tracer_type == "aaa", "For auto-correlation, tracer_type must be 'aaa'"


def validate_poles(poles):
    """
    Validate the poles input.
    Raise ValueError if the input is not valid.
    1. All elements must be non-negative integers.
    2. No duplicate values.
    3. If more than one value, they must be sorted in ascending order.
    """
    if not all(isinstance(p, int) and p >= 0 for p in poles):
        raise ValueError("All elements in 'poles' must be non-negative integers.")
    if len(poles) != len(set(poles)):
        raise ValueError("'poles' contains duplicate values.")
    if len(poles) > 1 and poles != sorted(poles):
        raise ValueError(
            "'poles' must be sorted in ascending order if it contains more than one value."
        )


def validate_sugi_poles(angu_config, geometry):
    """
    Validate the angu_config input for Sugiyama bispectrum estimator.
    Raise ValueError if the input is not valid.
    1. All ell elements must satisfy quantum angular number conditions.
    2. ell1 >= ell2
    3. The sum of the three ell elements must be even.
    """
    ell1, ell2, L = angu_config
    if ell1 < ell2:
        raise ValueError(
            "In 'angu_config', ell1 must be greater than or equal to ell2."
        )
    if (ell1 + ell2 + L) % 2 != 0:
        raise ValueError("The sum of ell1, ell2, and L in 'angu_config' must be even.")
    if L % 2 != 0:
        raise ValueError("L must be even.")
    if not (ell1 - ell2 <= L <= ell1 + ell2):
        raise ValueError(
            "The values in 'angu_config' do not satisfy the triangle condition."
        )
    if geometry not in ["box-like", "survey-like"]:
        raise ValueError("geometry must be either 'box-like' or 'survey-like'.")


def Cubic_Check(value, value_name, value_type):
    assert (
        isinstance(value, (list, tuple)) and len(value) == 3
    ), f"{value_name} must be a list or tuple of three elements"
    assert all(
        isinstance(x, value_type) and x > 0 for x in value
    ), f"All elements in {value_name} must be positive {value_type.__name__}s"
    assert len(set(value)) == 1, f"All elements in {value_name} must be equal"


def validate_boolean_fields(config):
    bool_fields = [
        "interlaced",
        "compensation",
        "use_fast_mode",
        "apply_rsd",
        "use_parent_dir",
    ]

    for field in bool_fields:
        if field not in config:
            continue
        if not isinstance(config[field], (bool, np.bool_)):
            raise TypeError(
                f"Config field '{field}' must be a boolean (True/False), got "
                f"{type(config[field]).__name__}: {config[field]!r}"
            )


def validate_config(config):
    if "shotnoise-mode" in config and "shotnoise_mode" not in config:
        config["shotnoise_mode"] = config.pop("shotnoise-mode")

    assert config["sampler"] in [
        "ngp",
        "cic",
        "pcs",
        "tsc",
    ], "sampler should be 'ngp', 'cic', 'pcs', or 'tsc'"
    Cubic_Check(config["nmesh"], "nmesh", int)
    Cubic_Check(config["boxsize"], "boxsize", (float, int))

    shotnoise_mode = config.setdefault("shotnoise_mode", "ana")
    if shotnoise_mode not in ["ana", "fft", "both"]:
        raise ValueError("shotnoise_mode must be either 'ana', 'fft', or 'both'")
    validate_boolean_fields(config)


def catalog_check(catalogs, geometry, correlation_mode, statistic, tracer_type=None):
    assert catalogs["data_a"] is not None, "data_a catalog must be provided"
    if statistic in ["pk", "bk_sco", "bk_sugi"]:
        if correlation_mode == "auto":
            if geometry == "survey-like":
                assert (
                    catalogs["randoms_a"] is not None
                ), "randoms_a catalog must be provided for survey-like auto-correlation"
        elif correlation_mode == "cross":
            assert (
                catalogs["data_b"] is not None
            ), "data_b catalog must be provided for cross-correlation"
            if geometry == "survey-like":
                assert (
                    catalogs["randoms_a"] is not None
                ), "randoms_a catalog must be provided for survey-like cross-correlation"
                assert (
                    catalogs["randoms_b"] is not None
                ), "randoms_b catalog must be provided for survey-like cross-correlation"
    if tracer_type == "abc" and statistic in ["bk_sco", "bk_sugi"]:
        assert (
            catalogs.get("data_c") is not None
        ), "data_c catalog must be provided for tracer_type 'abc' in bispectrum"
        if geometry == "survey-like":
            assert (
                catalogs.get("randoms_c") is not None
            ), "randoms_c catalog must be provided for tracer_type 'abc' in survey-like bispectrum"

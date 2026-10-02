import numpy as np
import pytest

from line_clearance.catenary import Catenary, Conductor, fit_catenary


def test_from_endpoints_passes_through_both_supports():
    cat = Catenary.from_endpoints(250.0, 120.0, 131.0, 1200.0)
    assert cat.z(0.0) == pytest.approx(120.0, abs=1e-6)
    assert cat.z(250.0) == pytest.approx(131.0, abs=1e-6)


def test_sag_matches_parabolic_approximation_on_level_span():
    length, c = 240.0, 1500.0
    cat = Catenary.from_endpoints(length, 50.0, 50.0, c)
    assert cat.sag(length) == pytest.approx(length**2 / (8 * c), rel=0.01)


def test_fit_recovers_the_curve_from_noisy_points():
    rng = np.random.default_rng(3)
    truth = Catenary.from_endpoints(260.0, 110.0, 118.0, 1100.0)
    l = rng.uniform(0, 260.0, 800)
    z = truth.z(l) + rng.normal(0, 0.03, l.size)
    z[:10] += 3.0  # a few returns from something else
    fitted, rmse = fit_catenary(l, z)
    grid = np.linspace(0, 260.0, 50)
    assert np.abs(fitted.z(grid) - truth.z(grid)).max() < 0.05
    assert fitted.sag(260.0) == pytest.approx(truth.sag(260.0), abs=0.05)
    assert rmse < 0.5


def test_fit_rejects_points_that_do_not_hang():
    l = np.linspace(0, 100, 50)
    with pytest.raises(ValueError):
        fit_catenary(l, 20.0 - 0.001 * (l - 50.0) ** 2)
    with pytest.raises(ValueError):
        fit_catenary(l[:3], np.ones(3))


def test_design_condition_keeps_supports_and_lowers_the_wire():
    cat = Catenary.from_endpoints(250.0, 120.0, 124.0, 1000.0)
    wire = Conductor((0.0, 0.0), (1.0, 0.0), 250.0, cat)
    flown = wire.sample(1.0)
    hot = wire.sample(1.0, sag_factor=1.25)
    np.testing.assert_allclose(hot[[0, -1]], flown[[0, -1]], atol=1e-6)
    assert (hot[1:-1, 2] < flown[1:-1, 2]).all()
    flown_sag = (np.linspace(120.0, 124.0, len(flown)) - flown[:, 2]).max()
    hot_sag = (np.linspace(120.0, 124.0, len(hot)) - hot[:, 2]).max()
    assert hot_sag == pytest.approx(1.25 * flown_sag, rel=0.02)


def test_blowout_swings_the_wire_sideways_around_the_chord():
    cat = Catenary.from_endpoints(200.0, 100.0, 100.0, 800.0)
    wire = Conductor((0.0, 0.0), (1.0, 0.0), 200.0, cat)
    still, blown = wire.sample(1.0), wire.sample(1.0, blowout_deg=30.0)
    mid = len(still) // 2
    sag = 100.0 - still[mid, 2]
    assert blown[mid, 1] == pytest.approx(sag * np.sin(np.deg2rad(30.0)), rel=1e-3)
    assert blown[mid, 2] == pytest.approx(100.0 - sag * np.cos(np.deg2rad(30.0)), rel=1e-3)
    np.testing.assert_allclose(blown[0], still[0], atol=1e-9)
    assert len(wire.envelope(1.0, 1.25, 20.0)) == 5 * len(still)

# -*- coding: utf-8 -*-
"""
Tests for ExoMol broadening species support (issue #602).

Verifies that changing the broadening partner does not affect the
total absorbance (integrated intensity), only the line shape.
"""

import numpy as np
import pytest


@pytest.mark.needs_connection
def test_exomol_broadening_species_abscoeff():
    """Test that changing broadening species does not change total absorbance.

    As requested by minouHub in PR #911:
    Changing the broadening partner (e.g. He vs air) only affects line
    widths, not the total integrated absorption coefficient.

    References
    ----------
    - Issue #602: https://github.com/radis/radis/issues/602
    - PR #911: https://github.com/radis/radis/pull/911
    """
    from radis import calc_spectrum

    # Compute spectrum with He broadening
    s_broad_He = calc_spectrum(
        2000,
        2100,
        molecule="CO2",
        isotope="1",
        pressure=0.01,
        Tgas=700,
        mole_fraction=1,
        databank="exomol",
        broadening_species="He",
        verbose=False,
    )

    # Compute spectrum with air broadening (default)
    s_broad_air = calc_spectrum(
        2000,
        2100,
        molecule="CO2",
        isotope="1",
        pressure=0.01,
        Tgas=700,
        mole_fraction=1,
        databank="exomol",
        broadening_species="air",
        verbose=False,
    )

    # Total absorbance must be conserved regardless of broadening partner
    # (broadening only redistributes intensity across wavenumbers)
    assert np.isclose(
        s_broad_He.get_integral("abscoeff"),
        s_broad_air.get_integral("abscoeff"),
        rtol=0.001,
    ), (
        "Total absorbance changed when switching broadening species from 'air' to 'He'. "
        "Broadening should only affect line shape, not total integrated intensity."
    )


@pytest.mark.needs_connection
def test_exomol_broadening_species_H2():
    """Test H2 broadening (relevant for gas giant atmospheres)."""
    from radis import calc_spectrum

    s_broad_H2 = calc_spectrum(
        2000,
        2100,
        molecule="CO2",
        isotope="1",
        pressure=0.01,
        Tgas=700,
        mole_fraction=1,
        databank="exomol",
        broadening_species="H2",
        verbose=False,
    )

    s_broad_air = calc_spectrum(
        2000,
        2100,
        molecule="CO2",
        isotope="1",
        pressure=0.01,
        Tgas=700,
        mole_fraction=1,
        databank="exomol",
        broadening_species="air",
        verbose=False,
    )

    assert np.isclose(
        s_broad_H2.get_integral("abscoeff"),
        s_broad_air.get_integral("abscoeff"),
        rtol=0.001,
    )


@pytest.mark.needs_connection
def test_exomol_broadening_species_default_is_air():
    """Test that default broadening species is 'air' (backward compatible)."""
    from radis import calc_spectrum

    # Without broadening_species (default)
    s_default = calc_spectrum(
        2000,
        2100,
        molecule="CO2",
        isotope="1",
        pressure=0.01,
        Tgas=700,
        mole_fraction=1,
        databank="exomol",
        verbose=False,
    )

    # With explicit broadening_species='air'
    s_air = calc_spectrum(
        2000,
        2100,
        molecule="CO2",
        isotope="1",
        pressure=0.01,
        Tgas=700,
        mole_fraction=1,
        databank="exomol",
        broadening_species="air",
        verbose=False,
    )

    # Both should give identical results
    assert np.isclose(
        s_default.get_integral("abscoeff"),
        s_air.get_integral("abscoeff"),
        rtol=1e-10,
    ), "Default broadening should be identical to explicit 'air' broadening."


def test_exomol_broadening_species_invalid():
    """Test that invalid broadening_species raises a clear ValueError."""
    from radis.io.exomol import fetch_exomol

    with pytest.raises(ValueError, match="broadening_species"):
        fetch_exomol(
            "CO2",
            broadening_species="invalid_species",
        )


if __name__ == "__main__":
    test_exomol_broadening_species_abscoeff()
    test_exomol_broadening_species_H2()
    test_exomol_broadening_species_default_is_air()
    test_exomol_broadening_species_invalid()
    print("All tests passed!")

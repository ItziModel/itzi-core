"""
Copyright (C) 2017-2026 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""

from enum import StrEnum


class DefaultValues:
    """Default config values"""

    # Threshold for determining the flow equation (m)
    HFMIN = 0.005
    # Coefficient applied to time-step calculation
    CFL = 0.7
    # Damping weighting coefficient
    THETA = 0.9
    # Standard gravity (m/s²)
    G = 9.80665
    # Maximum time-step duration (s)
    DTMAX = 5.0
    # Slope threshold (m/m). Value above which the GMS is applied
    SLOPE_THRESHOLD = 0.8
    # Maximum slope (m/m). Max value of slope to use with GMS
    MAX_SLOPE = SLOPE_THRESHOLD
    # Hydrology time step (s)
    DTINF = 60.0
    # Maximum continuity error allowed
    MAX_ERROR = 0.05
    # maximum Froude number. Not used yet.
    FRMAX = 1.0
    # coefficients taken from Rubinato et al. (2017)
    # http://doi.org/10.1016/j.jhydrol.2017.06.024
    ORIFICE_COEFF = 0.167
    FREE_WEIR_COEFF = 0.54
    SUBMERGED_WEIR_COEFF = 0.056
    # coefficients for drainage coupling stability
    RELAXATION_FACTOR = 0.8
    DAMPING_FACTOR = 0.5


class TemporalType(StrEnum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"


class InfiltrationModelType(StrEnum):
    NULL = "null"
    CONSTANT = "constant"
    GREEN_AMPT = "green-ampt"

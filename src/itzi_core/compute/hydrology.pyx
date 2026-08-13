"""
Copyright (C) 2026 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""
cimport cython
from cython.parallel cimport prange
from libc.math cimport INFINITY, isinf

ctypedef cython.floating DTYPE_t
cdef float PI = 3.1415926535898


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def apply_hydrology(
    DTYPE_t[:, :] arr_rain,
    DTYPE_t[:, :] arr_inf,
    DTYPE_t[:, :] arr_capped_losses,
    DTYPE_t[:, :] arr_h,
    DTYPE_t[:, :] arr_eff_precip,
    DTYPE_t dt,
):
    """Cap candidate water-removal rates and calculate effective rain.

    Available water is the depth present at the start of the hydrology event;
    current-event rainfall is deliberately excluded. If candidate infiltration
    and losses exceed that availability, both are scaled by the same factor.
    The candidate arrays are overwritten in place so the surface equation,
    report accumulators, mean maps, and Green-Ampt commit all use the same rates.

    The final effective-precipitation cap is defensive. For finite non-negative
    candidates, the combined removal cap should already prevent a negative depth.
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t available_rate, candidate_inf, candidate_losses, candidate_total
    cdef DTYPE_t scale, applied_inf, applied_losses, applied_total, excess
    cdef DTYPE_t hydro_raw, losses_limit
    rmax = arr_rain.shape[0]
    cmax = arr_rain.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            candidate_inf = arr_inf[r, c]
            candidate_losses = arr_capped_losses[r, c]
            candidate_total = candidate_inf + candidate_losses
            available_rate = arr_h[r, c] / dt
            if candidate_total > 0 and candidate_total > available_rate:
                # An infinite Green-Ampt candidate represents the F=0 limit.
                # Allocate finite available water without evaluating 0 * inf.
                if isinf(candidate_inf):
                    if isinf(candidate_losses):
                        applied_inf = available_rate / 2
                        applied_losses = available_rate - applied_inf
                    else:
                        applied_inf = available_rate
                        applied_losses = 0
                elif isinf(candidate_losses):
                    applied_inf = 0
                    applied_losses = available_rate
                else:
                    scale = available_rate / candidate_total
                    applied_inf = candidate_inf * scale
                    applied_losses = candidate_losses * scale
            else:
                applied_inf = candidate_inf
                applied_losses = candidate_losses

            # Keep the stored removal total within availability after dtype rounding.
            applied_total = applied_inf + applied_losses
            if applied_total > available_rate:
                excess = applied_total - available_rate
                if applied_losses >= excess:
                    applied_losses = applied_losses - excess
                else:
                    applied_inf = applied_inf - (excess - applied_losses)
                    applied_losses = 0
            arr_inf[r, c] = applied_inf
            arr_capped_losses[r, c] = applied_losses

            hydro_raw = arr_rain[r, c] - applied_inf - applied_losses
            losses_limit = - arr_h[r, c] / dt
            arr_eff_precip[r, c] = max(losses_limit, hydro_raw)


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def infiltration_user(
    DTYPE_t[:, :] arr_h,
    DTYPE_t[:, :] arr_inf_in,
    DTYPE_t[:, :] arr_inf_out,
    DTYPE_t dt
):
    """Copy a user-defined rate for later capping with infiltration.

    This kernel does not inspect available water. ``apply_hydrology`` performs
    the availability check after both candidate removal rates are known.
    """
    cdef int rmax, cmax, r, c

    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            arr_inf_out[r, c] = arr_inf_in[r, c]


@cython.wraparound(False)  # Disable negative index check
@cython.cdivision(True)  # Don't check division by zero
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def infiltration_ga(
    DTYPE_t[:, :] arr_h,
    DTYPE_t[:, :] arr_eff_por,
    DTYPE_t[:, :] arr_pressure,
    DTYPE_t[:, :] arr_conduct,
    DTYPE_t[:, :] arr_inf_amount,
    DTYPE_t[:, :] arr_water_soil_content,
    DTYPE_t[:, :] arr_inf_out,
    DTYPE_t dt
):
    """Calculate a Green-Ampt candidate without updating cumulative state.

    The candidate is ``K * (1 + delta_theta * (psi + h) / F)``, where ``F``
    is cumulative applied infiltration. A positive head term with ``F == 0``
    has an infinite limiting rate; ``apply_hydrology`` safely limits it to the
    available surface water. ``commit_infiltration`` updates ``F`` later.
    """
    cdef int rmax, cmax, r, c
    cdef DTYPE_t infrate, avail_porosity, poros_cappress, conduct
    rmax = arr_h.shape[0]
    cmax = arr_h.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            conduct = arr_conduct[r, c]
            avail_porosity = max(arr_eff_por[r, c] - arr_water_soil_content[r, c], 0)
            poros_cappress = avail_porosity * (arr_pressure[r, c] + arr_h[r, c])
            if arr_inf_amount[r, c] > 0:
                infrate = conduct * (1 + (poros_cappress / arr_inf_amount[r, c]))
            elif conduct > 0 and poros_cappress > 0:
                infrate = INFINITY
            else:
                infrate = conduct
            arr_inf_out[r, c] = infrate


@cython.wraparound(False)  # Disable negative index check
@cython.boundscheck(False)  # turn off bounds-checking for entire function
def commit_infiltration(
    DTYPE_t[:, :] arr_inf_amount,
    DTYPE_t[:, :] arr_applied_inf,
    DTYPE_t dt,
):
    """Advance Green-Ampt state using ``F += applied_rate * dt``.

    This must run after ``apply_hydrology`` has replaced the candidate rate
    with the rate accepted by the shared infiltration-and-losses cap.
    """
    cdef int rmax, cmax, r, c
    rmax = arr_inf_amount.shape[0]
    cmax = arr_inf_amount.shape[1]
    for r in prange(rmax, nogil=True):
        for c in range(cmax):
            arr_inf_amount[r, c] += arr_applied_inf[r, c] * dt

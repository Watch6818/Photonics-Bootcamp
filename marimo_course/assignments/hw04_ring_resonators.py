#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "marimo>=0.18.0",
#   "pyzmq",
#   "simphony==0.7.3",
#   "jax[cpu]",
#   "sax",
#   "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from _assignment_template import _ensure_lessons_on_path, load_lesson_template

    from textwrap import dedent as _dedent

    _ensure_lessons_on_path()
    inject_css, _make_doc_helpers, _make_health_refresh_button, header = load_lesson_template()
    inject_css(mo)

    header(
        mo,
        title="HW04 — Ring resonators (Simphony + design)",
        subtitle=(
            "Simulate ring resonators, measure FSR and Q, and design an add-drop ring."
        ),
        badges=["Week 4", "Homework", "Rings", "Simphony"],
        toc=[
            ("Overview", "overview"),
            ("Part A — Simulate 3 radii + measure FSR", "part-a"),
            ("Part B — Design for target FSR + estimate m", "part-b"),
            ("Part C — Add-drop ring + critical coupling + Q", "part-c"),
            ("Part D — Layout add-drop filter + DRC", "part-d"),
            ("Submission", "submit"),
        ],
        build="2026-01-28",
    )

    mo.callout(mo.md("Problem set (no solutions)."), kind="info")

    mo.md(
        _dedent(
            r"""
            <a id="overview"></a>
            ## Overview

            Four tasks:
            1. Simulate a ring resonator (3 radii) and measure the FSR.
            2. Design a ring for a target FSR and estimate the mode number near 1550 nm.
            3. Design an add-drop ring, attempt critical coupling, and estimate Q.
            4. Lay out an add-drop filter with grating couplers and pass DRC.
            """
        ).strip()
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
            r"""
            <a id="part-a"></a>
            ## Part A — Simulate 3 radii, measure FSR

            Simulate an ideal all-pass ring resonator in Simphony for three radii.
            From the plots, measure the FSR near 1550 nm and compare to the analytic result from class.
            """
        ).strip()
    )
    return


@app.cell
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
            r"""
            ### Part A deliverables

            - Plot transmission vs wavelength for 3 radii.
            - Report measured FSR for each radius.
            - Report analytic FSR for each radius and percent error.

            Helpful imports:

            ```python
            from jax import config
            config.update("jax_enable_x64", True)

            import jax.numpy as jnp
            import sax
            from simphony.libraries import ideal
            import matplotlib.pyplot as plt
            ```
            """
        ).strip()
    )
    return


@app.cell
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
            r"""
            Record your results in a table (R, measured FSR, analytic FSR, percent error).
            """
        ).strip()
    )
    return


@app.cell
def _(mo):
    _output = None
    _simphony_error = ""
    _wl_um = None

    try:
        from io import BytesIO
        from jax import config
        config.update("jax_enable_x64", True) # 64-bit floats
        import jax.numpy as jnp
        import math
        import matplotlib.pyplot as plt
        import numpy as np
        import sax
        from simphony.libraries import ideal # Not siepic yet; use ideal models

        # All values in this block are the defaults given in
        # w04_ring_resonators.py
        _R = [13, 14, 15]
        _coupling = 0.05
        _neff = 2.34
        _ng = 4.0
        _loss_db_per_cm = 2.0 # ?

        def build_wl_array(lambda0=1550, span=20, points=800):
            """
            lambda0: Initial wavelength. nm
            span: Distance from initial to last wavelength in array. nm
            points: Number of points in grid
            """
            lambda0_um , span_um = lambda0 * 1e-3, span * 1e-3
            array = jnp.linspace(
                lambda0_um - 0.5 * span_um,
                lambda0_um + 0.5 * span_um,
                points
            )
            return array, lambda0_um
        _wl_um, _lambda0_um = build_wl_array()

        _ring_circuit, _ = sax.circuit(
            netlist={
                "instances": {
                    "dc": "coupler",      # Initiate coupler
                    "loop": "waveguide",  # Initiate ring waveguide; simphony
                                          # docs say it's a straight waveguide 
                },
                "connections": {
                    # Close the ring loop between the two ring-side coupler
                    # ports.
                    "dc,o2": "loop,o0",   # "NW" dc port to loop's first port
                    "loop,o1": "dc,o3",   # loop's second port to "NE" dc port 
                },
                "ports": {
                    "input": "dc,o0",     # input to "SW" dc port
                    "through": "dc,o1",   # through to "SE" dc port
                },
            },
            models={
                "coupler": ideal.coupler, # ? Tells what coupler is
                "waveguide": ideal.waveguide,
            },
        )

        def get_transmission(radius, sc):
            """
            radius: Radius of given ring. μm
            sc : Pass in the sax.circuit() object you made.
            """
            S = None
            L = 2.0 * math.pi * radius

            S = sc(
                wl=_wl_um,
                dc={
                    "coupling": _coupling, # fraction of power coupled
                    "loss": 0.0,
                    "phi": 0.5 * math.pi, # π/2 phase convention
                },
                loop={
                    "wl0": _lambda0_um,
                    "neff": _neff,
                    "ng": _ng,
                    "length": L,
                    "loss": _loss_db_per_cm,
                },
            )
            t_through = S["through", "input"]
            intensity = jnp.abs(t_through) ** 2
            return [t_through, intensity]
        t_list = [
            get_transmission(_R[0], _ring_circuit),
            get_transmission(_R[1], _ring_circuit),
            get_transmission(_R[2], _ring_circuit)
            ]

    except Exception as e:  # pragma: no cover
        _simphony_error = f"{type(e).__name__}: {e}"

    # Handling simphony_error
    if _simphony_error:
        import sys

        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        if (
            "No module named 'jax'" in _simphony_error
            or 'No module named "jax"' in _simphony_error
            or "No module named 'sax'" in _simphony_error
            or 'No module named "sax"' in _simphony_error
        ):
            _output = mo.callout(
                mo.md(
                    (
                        "Simphony ring failed because required dependencies are missing.\n\n"
                        f"- Current Python: `{py_ver}`\n"
                        "- This Simphony example uses `simphony.libraries.ideal`, which imports `jax` and `sax`.\n\n"
                        "How to fix:\n"
                        "- **Recommended:** use marimo sandbox mode in a Python version with JAX wheels (often 3.11/3.12; Python 3.13 may not work):\n"
                        "  `marimo edit --sandbox marimo_course/lessons/w04_ring_resonators.py`\n"
                        "  (If your `marimo` is running under Python 3.13, use a 3.11/3.12 env, e.g. `./.venv/bin/python -m marimo ...`.)\n"
                        "- **Local venv:** install deps (may require Python 3.11/3.12):\n"
                        "  `./.venv/bin/pip install \"jax[cpu]\" sax`\n"
                    )
                ),
                kind="warn",
            )
        else:
            _output = mo.md(f"Simphony ring failed: `{_simphony_error}`")

    else:
        # Plot as a PNG for consistent rendering in marimo.
        _wl_nm = np.array(_wl_um) * 1e3

        _y0 = np.array(t_list[0][1])
        _y1 = np.array(t_list[1][1])
        _y2 = np.array(t_list[2][1])

        _fig = plt.figure()
        plt.plot(_wl_nm, _y0, 'r', _wl_nm, _y1, 'g', _wl_nm, _y2, 'b')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Through transmission |S|^2")
        plt.title("Ideal all-pass ring (Simphony) "
                  f"for R = {_R[0]}, {_R[1]}, and {_R[2]} μm",
                  pad=20
                 )
        plt.legend([f'R = {_R[0]}',f'R = {_R[1]}', f'R = {_R[2]}'])
        _subtitle = (
            f"kappa={_coupling:.2f}, "
            f"neff={_neff:.2f}, ng={_ng:.2f}, "
            f"loss={_loss_db_per_cm:.2f} dB/cm"
            )

        plt.text(
                0.5,
                1.02,
                _subtitle,
                transform=plt.gca().transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        _fig.tight_layout()
        _buf = BytesIO()
        _fig.savefig(_buf, format="png", bbox_inches="tight")
        _buf.seek(0)
        plt.close(_fig)

        from scipy.signal import find_peaks, peak_widths
        def get_m_FSR(y): # Measured FSR
            peaks, _ = find_peaks(-y)
            return _wl_nm[peaks[1]] - _wl_nm[peaks[0]]
        def get_a_FSR(radius): # Analytic FSR
            L = 2 * math.pi * radius
            return (_lambda0_um * 10**3)**2 / (_ng * L * 10**3)
        _fsr_0 = get_m_FSR(_y0)
        _fsr_1 = get_m_FSR(_y1)
        _fsr_2 = get_m_FSR(_y2)
        _fsr_a_0 = get_a_FSR(_R[0])
        _fsr_a_1 = get_a_FSR(_R[1])
        _fsr_a_2 = get_a_FSR(_R[2])
        _err_0 = abs(_fsr_a_0 - _fsr_0)/_fsr_a_0 * 100
        _err_1 = abs(_fsr_a_1 - _fsr_1)/_fsr_a_1 * 100
        _err_2 = abs(_fsr_a_2 - _fsr_2)/_fsr_a_2 * 100
        _output = mo.vstack([
            mo.md("## Part A: Plots and Results"),
            mo.image(_buf),
            mo.md(f"**Measured FSR (R={_R[0]} μm):** {_fsr_0:.2f} nm"),
            mo.md(f"Analytic FSR (R={_R[0]} μm): {_fsr_a_0:.2f} nm"),
            mo.md(f"Percent Error: {_err_0:.2f}%"),
            mo.md(f"**Measured FSR (R={_R[1]} μm):** {_fsr_1:.2f} nm"),
            mo.md(f"Analytic FSR (R={_R[1]} μm): {_fsr_a_1:.2f} nm"),
            mo.md(f"Percent Error: {_err_1:.2f}%"),
            mo.md(f"**Measured FSR (R={_R[2]} μm):** {_fsr_2:.2f} nm"),
            mo.md(f"Analytic FSR (R={_R[2]} μm): {_fsr_a_2:.2f} nm"),
            mo.md(f"Percent Error: {_err_2:.2f}%")
        ])
    _output
    return (
        BytesIO,
        build_wl_array,
        find_peaks,
        jnp,
        math,
        np,
        peak_widths,
        plt,
        sax,
    )


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
            r"""
            <a id="part-b"></a>
            ## Part B — Design for target FSR

            Pick a target FSR near 1550 nm. Choose a ring radius that meets it (first pass),
            then estimate the mode number near 1550 nm (m).
            """
        ).strip()
    )
    return


@app.cell
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
            r"""
            ### Part B deliverables

            - Target FSR and chosen radius.
            - Estimated m near 1550 nm (value + nearest integer).
            """
        ).strip()
    )
    return


@app.cell
def _(math):
    def mode_number(FSR, neff, ng):
        """
        Returns a list containing in order:
        - target FSR
        - exact ring radius
        - exact mode number
        - mode number rounded to closest integer

        NEEDS TO GET FINALIZED FOR PRINTING OUT DELIVERABLES
        """
        lambda0 = 1550
        ring_radius = lambda0**2 / (FSR * ng * 2 * math.pi) # FSR eqn
        mode_num = neff * 2 * math.pi * ring_radius / lambda0
        return [FSR, ring_radius * 10**-3, mode_num, round(mode_num)]
    return


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
            r"""
            <a id="part-c"></a>
            ## Part C — Add-drop ring + critical coupling + Q

            Design an **add-drop ring resonator** and simulate both **through** and **drop** ports.

            Goals:
            - Adjust coupling to attempt **critical coupling** (largest on-resonance extinction in the through port).
            - Estimate the **loaded Q** from your simulated spectrum (from the linewidth).
            - Compare your measured Q to an analytic estimate from class.

            ### Part C deliverables

            - Through + drop spectra for your add-drop ring (axes labeled).
            - A short note describing how you tuned coupling and whether you achieved near-critical coupling.
            - Q from the plot and your analytic Q estimate (with assumptions stated).
            """
        ).strip()
    )
    return


@app.cell
def _(BytesIO, build_wl_array, find_peaks, jnp, mo, np, peak_widths, plt, sax):
    from functools import partial
    from simphony.libraries import siepic

    ## Perform simulation ##
    _wl_um = build_wl_array()[0]
    _pol = 'te'
    _coupling_length = 0
    _width = 500
    _thickness = 220
    _radius = 10
    _gap = 100


    ring_add_drop, _ = sax.circuit(
        netlist={
            "instances": {
                "top":    "half_ring",
                "bottom": "half_ring",
                "term": "terminator", # Escape light at terminator → no reflect
            },
            "connections": {
                "top,port_3":    "bottom,port_1",
                "bottom,port_3": "top,port_1",
                "term,port_1": "bottom,port_4"
            },
            "ports": {
                "in":      "top,port_4",
                "through": "top,port_2",
                "drop":    "bottom,port_2",
            },
        },
        models={
            "half_ring": partial(
                siepic.half_ring,
                wl=_wl_um,
                pol=_pol,
                coupling_length=_coupling_length,
                width=_width,
                thickness=_thickness,
                radius=_radius,
                gap=_gap,
                ),
            "terminator": partial(siepic.terminator, pol=_pol)
        },
    )

    _S = ring_add_drop(wl=_wl_um)
    i_through = jnp.abs(_S["through", "in"]) ** 2 # Intensity array of through port
    i_drop = jnp.abs(_S["drop", "in"]) ** 2

    ## Plot results ##
    _wl_nm = _wl_um * 1e3
    yd = np.array(i_drop)
    yt = np.array(i_through)

    _fig = plt.figure()
    plt.plot(_wl_nm, yt, 'r', _wl_nm, yd, 'g')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission |S|^2")
    plt.title("SiEPIC add-drop ring (Simphony)", pad=20)
    plt.legend(['Through Port', 'Drop Port'])
    _subtitle = (
            f"pol={_pol}, "
            f"coupling length={_coupling_length} μm, "
            f"width={_width} nm, "
            f"thickness={_thickness} nm, "
            f"radius={_radius} μm, "
            f"gap={_gap} nm"
            )
    plt.text(
        0.5,
        1.02,
        _subtitle,
        transform = plt.gca().transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        )
    _fig.tight_layout()
    _buf = BytesIO()
    _fig.savefig(_buf, format="png", bbox_inches="tight")
    _buf.seek(0)
    plt.close(_fig)

    ## Calculate Q ##
    def calculate_Q(wavelengths, intensity_array, has_dips=False):
        if has_dips:
            intensity_array = - intensity_array
        nm_per_index = wavelengths[1] - wavelengths[0]
        peaks, _ = find_peaks(intensity_array, prominence=0.125)
        widths = peak_widths(intensity_array, peaks)
        fwhm_nm = widths[0] * nm_per_index # widths[0]: np.array
        Q_vals = wavelengths[peaks] / fwhm_nm
        Q_avg = Q_vals.mean()
        return Q_vals, Q_avg
    Q = calculate_Q(_wl_nm, yd)
    print(Q)
    _output = mo.image(_buf)
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
            r"""
            <a id="part-d"></a>
            ## Part D — Layout add-drop filter + DRC

            Lay out an **add-drop ring filter** with **grating couplers** (input, through, drop).
            Export a GDS and make sure it passes all DRC checks.

            ### Part D deliverables

            - GDS of your add-drop filter with grating couplers.
            - A DRC report showing 0 items.
            - A screenshot of the layout.
            """
        ).strip()
    )
    return


@app.cell
def _(combiner_cell, combiner_gds, mo, splitter_cell, splitter_gds):
    # SOLUTION VALUES — update if your paths differ
    import pathlib as pathlib_params

    username = "Watch6818"
    repo_root = pathlib_params.Path(__file__).resolve().parents[3]
    ebeam_pdk_path = str(repo_root / "SiEPIC_EBeam_PDK")
    openebl_path = str(repo_root / "openEBL-2026-02")

    delta_length_um = 300.0
    length_x_um = 60.0
    length_y_um = 10.0

    ring_double = f"{ebeam_pdk_path}/klayout/EBeam/gds/EBeam/"

    gc_gds = f"{ebeam_pdk_path}/klayout/EBeam/gds/EBeam/ebeam_gc_te1550.gds"
    gc_cell = "ebeam_gc_te1550"



    export_gds = f"{openebl_path}/submissions/EBeam_{username}.gds"

    mo.md(
        f"- ΔL: `{delta_length_um}` µm\n"
        f"- length_x: `{length_x_um}` µm\n"
        f"- length_y: `{length_y_um}` µm\n"
        f"- splitter cell: `{splitter_cell}`\n"
        f"- combiner cell: `{combiner_cell}`\n"
        f"- GC cell: `{gc_cell}`\n"
        f"- export path: `{export_gds}`\n"
    )

    import pathlib as pathlib_hw
    import gdsfactory as gf
    from gdsfactory.add_ports import add_ports_from_markers_center
    from gdsfactory.read import import_gds
    from gdsfactory.port import auto_rename_ports_orientation

    gf.clear_cache() # Ensure components will be built fresh
    if hasattr(gf, "gpdk") and hasattr(gf.gpdk, "PDK"):
        gf.gpdk.PDK.activate() # Set current PDK to active PDK
    else:
        gf.pdk.get_generic_pdk().activate() # Activate generic PDK
    c = None # Gigantic variable that will be written to file

    xs = gf.cross_section.strip(layer=(1, 0), width=0.5) # Cross section

    # Check for GDS files
    splitter_path = pathlib_hw.Path(splitter_gds) # Create a Path object
    combiner_path = pathlib_hw.Path(combiner_gds)
    gc_path = pathlib_hw.Path(gc_gds)
    mo.stop(
        not splitter_path.exists(),
        mo.md(f"Error: Splitter GDS not found: `{splitter_path}`")
    )
    mo.stop(
        not combiner_path.exists(),
        mo.md(f"Error: Combiner GDS not found: `{combiner_path}`")
    )
    mo.stop(
        not gc_path.exists(),
        mo.md(f"Error: GC GDS not found: `{gc_path}`")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    from textwrap import dedent as _dedent

    mo.md(
        _dedent(
            r"""
            <a id="submit"></a>
            ## Submission

            Submit your plots/calculations (Parts A, B, C) and your layout artifacts (Part D).
            """
        ).strip()
    )
    return


if __name__ == "__main__":
    app.run()

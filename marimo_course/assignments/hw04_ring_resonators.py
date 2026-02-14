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
    S = None
    intensity = None
    import_seconds = None
    output = None
    simphony_error = ""
    wl_um = None

    try:
        import time

        t0 = time.time()
        from jax import config
        config.update("jax_enable_x64", True) # 64-bit floats
        import jax.numpy as jnp
        import math
        import matplotlib.pyplot as plt
        import sax
        from simphony.libraries import ideal # Not siepic yet; use ideal models
        import_seconds = time.time() - t0 # Time taken to import

        # All values in this block are the defaults given in
        # w04_ring_resonators.py
        R = [13, 14, 15]
        coupling = 0.05
        neff = 2.34
        ng = 4.0
        loss_db_per_cm = 2.0 # ?

        _lambda0_um = 1550 * 1e-3
        _span_um = 20 * 1e-3
        points = 800
        wl_um = jnp.linspace(
            _lambda0_um - 0.5 * _span_um,
            _lambda0_um + 0.5 * _span_um,
            points,
        )

        ring_circuit, _ = sax.circuit(
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

        def get_transmission(R):
            _R_um = R                       # One of our ring values in μm
            _L_um = 2.0 * math.pi * _R_um

            S = ring_circuit(
                wl=wl_um,
                dc={
                    "coupling": coupling, # fraction of power coupled
                    "loss": 0.0,
                    "phi": 0.5 * math.pi, # π/2 phase convention
                },
                loop={
                    "wl0": _lambda0_um,
                    "neff": neff,
                    "ng": ng,
                    "length": _L_um,
                    "loss": loss_db_per_cm,
                },
            )
            t_through = S["through", "input"]
            intensity = jnp.abs(t_through) ** 2 # What we plot using matplotlib
            return [t_through, intensity]
        t_list = [
            get_transmission(R[0]),
            get_transmission(R[1]),
            get_transmission(R[2])
            ]

    except Exception as e:  # pragma: no cover
        simphony_error = f"{type(e).__name__}: {e}"

    # Handling simphony_error
    if simphony_error:
        import sys

        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        if (
            "No module named 'jax'" in simphony_error
            or 'No module named "jax"' in simphony_error
            or "No module named 'sax'" in simphony_error
            or 'No module named "sax"' in simphony_error
        ):
            output = mo.callout(
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
            output = mo.md(f"Simphony ring failed: `{simphony_error}`")

    else:
        # Plot as a PNG for consistent rendering in marimo.
        import numpy as np
        from io import BytesIO

        import matplotlib.pyplot as plt

        wl_nm = np.array(wl_um) * 1e3

        y0 = np.array(t_list[0][1])
        y1 = np.array(t_list[1][1])
        y2 = np.array(t_list[2][1])

        fig = plt.figure()
        plt.plot(wl_nm, y0, 'r', wl_nm, y1, 'g', wl_nm, y2, 'b')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Through transmission |S|^2")
        plt.title("Ideal all-pass ring (Simphony) "
                  f"for R = {R[0]}, {R[1]}, and {R[2]} μm",
                  pad=20
                 )
        plt.legend([f'R = {R[0]}',f'R = {R[1]}', f'R = {R[2]}'])
        subtitle = (
                f"kappa={coupling:.2f}, "
                f"neff={neff:.2f}, ng={ng:.2f}, "
                f"loss={loss_db_per_cm:.2f} dB/cm"
            )

        plt.text(
                0.5,
                1.02,
                subtitle,
                transform = plt.gca().transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        from scipy.signal import find_peaks
        def get_m_FSR(y): # Measured FSR
            peaks, _ = find_peaks(-y)
            return wl_nm[peaks[1]] - wl_nm[peaks[0]]
        def get_a_FSR(R): # Analytic FSR
            L = 2 * math.pi * R
            return (_lambda0_um * 10**3)**2 / (ng * L * 10**3)
        fsr_0 = get_m_FSR(y0)
        fsr_1 = get_m_FSR(y1)
        fsr_2 = get_m_FSR(y2)
        fsr_a_0 = get_a_FSR(R[0])
        fsr_a_1 = get_a_FSR(R[1])
        fsr_a_2 = get_a_FSR(R[2])
        err_0 = abs(fsr_a_0 - fsr_0)/fsr_a_0 * 100
        err_1 = abs(fsr_a_1 - fsr_1)/fsr_a_1 * 100
        err_2 = abs(fsr_a_2 - fsr_2)/fsr_a_2 * 100
        output = mo.vstack([
            mo.image(buf),
            mo.md(f"**Measured FSR (R={R[0]} μm):** {fsr_0:.2f} nm"),
            mo.md(f"Analytic FSR (R={R[0]} μm): {fsr_a_0:.2f} nm"),
            mo.md(f"Percent Error: {err_0:.2f}%"),
            mo.md(f"**Measured FSR (R={R[1]} μm):** {fsr_1:.2f} nm"),
            mo.md(f"Analytic FSR (R={R[1]} μm): {fsr_a_1:.2f} nm"),
            mo.md(f"Percent Error: {err_1:.2f}%"),
            mo.md(f"**Measured FSR (R={R[2]} μm):** {fsr_2:.2f} nm"),
            mo.md(f"Analytic FSR (R={R[2]} μm): {fsr_a_2:.2f} nm"),
            mo.md(f"Percent Error: {err_2:.2f}%")
        ])
    output
    return math, neff, ng


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
def _(math, neff, ng):
    def mode_number(FSR):
        """
        Returns a list containing in order:
        - target FSR
        - exact ring radius
        - exact mode number
        - mode number rounded to the closest integer

        NEEDS TO GET FINALIZED FOR PRINTING OUT DELIVERABLES
        """
        _lambda0 = 1550
        ring_radius = _lambda0**2 / (FSR * ng * 2 * math.pi) # FSR eqn
        mode_num = neff * 2 * math.pi * ring_radius / _lambda0
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
def _():
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

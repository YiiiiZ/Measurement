# resonator_spec_funcs.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# LabOne Q / Zurich Instruments imports (adjust if your env differs)
from laboneq.dsl import Experiment, ExperimentSignal, AcquisitionType, LinearSweepParameter, Calibration, SignalCalibration
from laboneq.dsl.calibration import Oscillator, ModulationType
from laboneq.contrib.example_helpers.plotting.plot_helpers import show_pulse_sheet

# -------------------------- #
# Construction & parameters  #
# -------------------------- #
def init_context(
    *,
    device_setup,
    session,
    center_freq: float,
    start_freq: float,
    stop_freq: float,
    num_points: int,
    integration_time: float,
    num_averages: int,
    measure_range: float,
    measure_amp: float,
    acquire_range: float,
    outdir: str = "resonator_results",
    results_prefix: str | None = None,
) -> dict:
    ctx = {}
    ctx["device_setup"] = device_setup
    ctx["session"] = session

    # Coarse sweep params (overall_spec)
    ctx["center_freq"] = float(center_freq)
    ctx["start_freq"] = float(start_freq)
    ctx["stop_freq"] = float(stop_freq)
    ctx["num_points"] = int(num_points)
    ctx["integration_time"] = float(integration_time)
    ctx["num_averages"] = int(num_averages)

    # Signal calibration
    ctx["measure_range"] = float(measure_range)
    ctx["measure_amp"] = float(measure_amp)
    ctx["acquire_range"] = float(acquire_range)

    # Derived: measure total power in dBm (requested)
    amp_for_db = float(np.clip(ctx["measure_amp"], 1e-12, None))
    ctx["measure_total_dbm"] = ctx["measure_range"] + 20.0 * np.log10(amp_for_db)

    ctx["qubit"] = str(qubit)

    # I/O and naming
    ctx["outdir"] = Path(outdir)
    ctx["outdir"].mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ctx["results_prefix"] = results_prefix or f"res_spec_{ts}"

    # --- Stage 1 containers ---
    ctx["_overall_freq_sweep"] = None
    ctx["_overall_exp"] = None
    ctx["_overall_compiled"] = None
    ctx["_overall_result"] = None

    ctx["freq_list"] = None         # Hz (coarse)
    ctx["amp"] = None               # complex (coarse)
    ctx["amp_dbm"] = None
    ctx["phase_rad"] = None
    ctx["phase_rad_detrended"] = None
    ctx["dips_freq"] = None         # Hz
    ctx["dips_amp_dbm"] = None

    # --- Stage 2 containers ---
    ctx["individual_runs"] = []     # list of dicts, one per target

    return ctx

# -------------------------- #
# Shared building blocks     #
# -------------------------- #
def _make_signal_map(ctx: dict):
    return {
        "measure": ctx["device_setup"].logical_signal_groups[ctx["qubit"]].logical_signals["measure_line"],
        "acquire": ctx["device_setup"].logical_signal_groups[ctx["qubit"]].logical_signals["acquire_line"],
    }

def _make_calibration(ctx: dict, frequency_source):
    calib = Calibration()
    calib["measure"] = SignalCalibration(
        oscillator=Oscillator(uid="qa_osc_0", frequency=frequency_source, modulation_type=ModulationType.HARDWARE),
        local_oscillator=Oscillator(uid="qa_lo", frequency=ctx["center_freq"]),
        range=ctx["measure_range"],
        amplitude=ctx["measure_amp"],
    )
    calib["acquire"] = SignalCalibration(range=ctx["acquire_range"])
    return calib

def _build_experiment(ctx: dict, freq_sweep, uid="Resonator Spectroscopy", integration_time=None, num_averages=None):
    exp = Experiment(uid=uid, signals=[ExperimentSignal("measure"), ExperimentSignal("acquire")])
    with exp.acquire_loop_rt(
        uid="shots",
        count=pow(2, int(num_averages if num_averages is not None else ctx["num_averages"])),
        acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        with exp.sweep(uid="res_freq", parameter=freq_sweep):
            with exp.section(uid="spectroscopy"):
                exp.acquire(
                    signal="acquire",
                    handle="res_spec",
                    length=float(integration_time if integration_time is not None else ctx["integration_time"]),
                )
            with exp.section(uid="relax", length=120e-6):
                exp.reserve(signal="measure")
                exp.reserve(signal="acquire")
    return exp

# ========================== #
# STAGE 1: overall_spec_*    #
# ========================== #
def overall_spec_run(ctx: dict, save_pulse_sheet: bool = False):
    """(overall_spec) Run a single coarse sweep across [start, stop] relative to LO."""
    ctx["_overall_freq_sweep"] = LinearSweepParameter(
        uid=f"{ctx['qubit']}_res_freq_overall",
        start=ctx["start_freq"],
        stop=ctx["stop_freq"],
        count=ctx["num_points"],
        axis_name="Frequency [Hz]",
    )

    ctx["_overall_exp"] = _build_experiment(
        ctx, ctx["_overall_freq_sweep"], uid="Resonator Spectroscopy (overall_spec)"
    )
    ctx["_overall_exp"].set_calibration(_make_calibration(ctx, ctx["_overall_freq_sweep"]))
    ctx["_overall_exp"].set_signal_map(_make_signal_map(ctx))

    ctx["_overall_compiled"] = ctx["session"].compile(ctx["_overall_exp"])

    if save_pulse_sheet:
        Path("Pulse_Sheets").mkdir(parents=True, exist_ok=True)
        show_pulse_sheet(str(ctx["outdir"] / f"{ctx['results_prefix']}_overall_spec_pulsesheet"), ctx["_overall_compiled"])

    ctx["_overall_result"] = ctx["session"].run(ctx["_overall_compiled"])
    return ctx["_overall_result"]

def overall_spec_analyze(ctx: dict, dips_to_take: int = 6):
    """(overall_spec) Find dips, compute dBm & detrended phase. No fitting here."""
    if ctx["_overall_result"] is None:
        raise RuntimeError("overall_spec_run() first.")

    amp = np.asarray(ctx["_overall_result"].acquired_results["res_spec"].data).ravel()
    freqs = np.asarray(ctx["_overall_result"].acquired_results["res_spec"].axis[0]).ravel()

    freq_list = freqs.astype(float) + float(ctx["center_freq"])
    amplitude = np.abs(amp)
    power_W = (amplitude**2) / 50.0
    amp_dbm = 10.0 * np.log10(power_W / 1e-3)

    order = np.argsort(freq_list)
    freq_list = freq_list[order]
    amp = amp[order]
    amp_dbm = amp_dbm[order]

    phase_rad = np.unwrap(np.arctan2(amp.imag, amp.real))

    f_ghz = freq_list / 1e9
    f0 = f_ghz - f_ghz.mean()

    mask_fin = np.isfinite(f0) & np.isfinite(phase_rad)
    x = f0[mask_fin]
    y = phase_rad[mask_fin]

    if x.size >= 2 and np.ptp(x) > 0:
        try:
            slope_c, intercept = np.polyfit(x, y, 1)
        except np.linalg.LinAlgError:
            slope_c, intercept = 0.0, float(np.median(y))
    else:
        slope_c, intercept = 0.0, float(np.median(y))

    phase_rad_detrended = phase_rad - (slope_c * f0 + intercept)

    def _cand_minima(y):
        return np.where((y[1:-1] <= y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1

    try:
        from scipy.signal import find_peaks
        inv = -amp_dbm
        peaks, _ = find_peaks(inv, prominence=0.2, distance=max(1, len(inv)//200))
        cand_idx = peaks if len(peaks) else _cand_minima(amp_dbm)
    except Exception:
        cand_idx = _cand_minima(amp_dbm)
    if len(cand_idx) == 0:
        cand_idx = np.array([int(np.argmin(amp_dbm))])

    take = min(dips_to_take, len(cand_idx))
    deep_order = np.argsort(amp_dbm[cand_idx])[:take]
    dips_idx = cand_idx[deep_order]

    # quadratic refinement
    dips_freq, dips_amp = [], []
    for i in dips_idx:
        f_est, a_est = freq_list[i], amp_dbm[i]
        if 0 < i < len(freq_list)-1:
            xq = freq_list[i-1:i+2]
            yq = amp_dbm[i-1:i+2]
            qa, qb, qc = np.polyfit(xq, yq, 2)
            if qa > 0:
                f_v = -qb / (2 * qa)
                if xq[0] <= f_v <= xq[2]:
                    f_est = f_v
                    a_est = qa * f_v**2 + qb * f_v + qc
        dips_freq.append(f_est)
        dips_amp.append(a_est)

    dips_freq = np.array(dips_freq)
    dips_amp = np.array(dips_amp)
    srt = np.argsort(dips_freq)
    dips_freq = dips_freq[srt]
    dips_amp = dips_amp[srt]

    # store
    ctx["freq_list"] = freq_list
    ctx["amp"] = amp
    ctx["amp_dbm"] = amp_dbm
    ctx["phase_rad"] = phase_rad
    ctx["phase_rad_detrended"] = phase_rad_detrended
    ctx["dips_freq"] = dips_freq
    ctx["dips_amp_dbm"] = dips_amp

    # save CSVs (overall_spec)
    np.savetxt(
        ctx["outdir"] / f"{ctx['results_prefix']}_overall_spec_amplitude.csv",
        np.column_stack([ctx["freq_list"], ctx["amp_dbm"]]),
        delimiter=",",
        header=f"freq_Hz, amplitude_dBm\n# measure_total_dBm={ctx['measure_total_dbm']:.6f}"
    )
    np.savetxt(
        ctx["outdir"] / f"{ctx['results_prefix']}_overall_spec_phase.csv",
        np.column_stack([ctx["freq_list"], ctx["phase_rad_detrended"], ctx["phase_rad"]]),
        delimiter=",",
        header="freq_Hz, phase_rad_detrended, phase_rad"
    )

    return {
        "freq_list_Hz": ctx["freq_list"],
        "amp_complex": ctx["amp"],
        "amp_dbm": ctx["amp_dbm"],
        "phase_rad": ctx["phase_rad"],
        "phase_rad_detrended": ctx["phase_rad_detrended"],
        "dips_freq_Hz": ctx["dips_freq"],
        "dips_amp_dbm": ctx["dips_amp_dbm"],
        "measure_total_dBm": ctx["measure_total_dbm"],
    }

def overall_spec_plot(ctx: dict, save_png: bool = True):
    """(overall_spec) Plot coarse amplitude/phase with dip markers."""
    if any(x is None for x in [ctx["freq_list"], ctx["amp_dbm"], ctx["phase_rad_detrended"], ctx["dips_freq"]]):
        raise RuntimeError("overall_spec_analyze() first.")
    fGHz = ctx["freq_list"] / 1e9
    dipsGHz = ctx["dips_freq"] / 1e9

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios': [1.5, 1.0]}
    )
    ax1.plot(fGHz, ctx["amp_dbm"], "-")
    ax1.set_ylabel("Amplitude [dBm]")
    ax1.set_title(f"Resonator Spectroscopy — {ctx['measure_total_dbm']:.1f} dBm")

    for f_min_GHz, a_min in zip(dipsGHz, ctx["dips_amp_dbm"]):
        ax1.scatter([f_min_GHz], [a_min], marker="x", s=90, zorder=5)
        ax1.axvline(x=f_min_GHz, linestyle="--", alpha=0.35)
        ax1.annotate(f"{f_min_GHz:.4f} GHz", (f_min_GHz, a_min),
                     textcoords="offset points", xytext=(6, 8), fontsize=8)

    ax2.plot(fGHz, ctx["phase_rad_detrended"], "-", color="tab:purple")
    for f_min_GHz in dipsGHz:
        ax2.axvline(x=f_min_GHz, linestyle="--", alpha=0.35)
    ax2.set_xlabel("Frequency [GHz]")
    ax2.set_ylabel("Phase [rad]")
    plt.tight_layout()

    png = None
    if save_png:
        png = ctx["outdir"] / f"{ctx['results_prefix']}_overall_spec.png"
        fig.savefig(png, dpi=600)
    print(f'minimum_freq, {dipsGHz}')
    return fig, {"png": str(png) if png else None}

# ============================== #
# STAGE 2: individual_spec_*     #
# ============================== #

# --- Notch-model (no cable delay) ---
def s21_notch_complex(f, fr, Ql, Qc_mag, phi, a_mag):
    """
    Complex S21(f) for a notch resonator (no delay/offset):
    S21(f) = a * [ 1 - (Ql/Qc_mag) * e^{i phi} / (1 + 2i Ql (f/fr - 1)) ]
    """
    x   = (f / fr) - 1.0
    num = (Ql / Qc_mag) * np.exp(1j * phi)
    den = 1.0 + 2j * Ql * x
    return a_mag * (1.0 - num / den)

def s21_notch_mag(f, fr, Ql, Qc_mag, phi, a_mag):
    return np.abs(s21_notch_complex(f, fr, Ql, Qc_mag, phi, a_mag))

def guess_notch_params(f_Hz, y_mag):
    """Heuristic initial guesses from |S21| dip."""
    hi = np.partition(y_mag, int(0.8*len(y_mag)))[int(0.8*len(y_mag)):]
    a0 = float(np.median(hi)) if len(hi) else float(np.median(y_mag))
    a0 = max(a0, 1e-12)

    i_min = int(np.argmin(y_mag))
    fr0  = float(f_Hz[i_min])
    ymin = float(y_mag[i_min])

    depth = np.clip(1.0 - ymin / a0, 1e-4, 0.99)

    span_idx = max(3, len(f_Hz)//40)
    left  = max(i_min - span_idx, 0)
    right = min(i_min + span_idx, len(f_Hz)-1)
    mid = ymin + 0.5*(a0 - ymin)
    try: left  = np.argmin(np.abs(y_mag[:i_min] - mid))
    except: pass
    try: right = i_min + np.argmin(np.abs(y_mag[i_min:] - mid))
    except: pass

    fwhm = max(f_Hz[right] - f_Hz[left], max(fr0 * 1e-6, np.finfo(float).eps))
    Ql0  = float(abs(fr0 / fwhm))
    Qc0  = float(Ql0 / depth)
    phi0 = 0.0

    fr0 = max(fr0, 1.0)
    Ql0 = float(np.clip(Ql0, 1e2, 1e9))
    Qc0 = float(np.clip(Qc0, 1e2, 1e12))
    a0  = float(np.clip(a0, 1e-12, 1e6))
    return fr0, Ql0, Qc0, phi0, a0

def fit_notch_magnitude(f_Hz, y_mag, guess=None):
    """
    Fit |S21| with the notch model; returns dict {fr, Ql, Qc_mag, phi, a, cov, method}.
    Magnitude-only fit (phase offset handled later).
    """
    if guess is None:
        guess = guess_notch_params(f_Hz, y_mag)

    try:
        from scipy.optimize import curve_fit
        lower = [min(f_Hz)*0.5, 1e2, 1e2, -np.pi, 1e-12]
        upper = [max(f_Hz)*1.5, 1e9, 1e12, np.pi, 1e6]
        popt, pcov = curve_fit(
            lambda f, fr, Ql, Qc_mag, phi, a: s21_notch_mag(f, fr, Ql, Qc_mag, phi, a),
            f_Hz, y_mag, p0=guess, bounds=(lower, upper), maxfev=30000
        )
        fr, Ql, Qc_mag, phi, a = map(float, popt)
        return {"fr": fr, "Ql": abs(Ql), "Qc_mag": abs(Qc_mag), "phi": phi, "a": abs(a), "cov": pcov, "method": "scipy"}
    except Exception:
        fr0, Ql0, Qc0, phi0, a0 = guess
        fr_grid  = np.linspace(fr0*0.999, fr0*1.001, 15)
        Ql_grid  = np.linspace(max(1e2, Ql0/4),  min(1e9,  Ql0*4),  15)
        Qc_grid  = np.linspace(max(1e2, Qc0/4),  min(1e12, Qc0*4), 12)
        phi_grid = np.linspace(-0.5*np.pi, 0.5*np.pi, 11)
        a_grid   = np.linspace(max(1e-12, a0/3),   min(1e6,  a0*3),  10)
        best = None; best_err = np.inf
        for fr_ in fr_grid:
            for Ql_ in Ql_grid:
                for Qc_ in Qc_grid:
                    for phi_ in phi_grid:
                        base = s21_notch_mag(f_Hz, fr_, Ql_, Qc_, phi_, 1.0)
                        for a_ in a_grid:
                            y_pred = a_ * (base / np.max(base))
                            err = np.mean((y_mag - y_pred)**2)
                            if err < best_err:
                                best_err = err; best = (fr_, Ql_, Qc_, phi_, a_)
        fr, Ql, Qc_mag, phi, a = best
        return {"fr": float(fr), "Ql": float(abs(Ql)), "Qc_mag": float(abs(Qc_mag)),
                "phi": float(phi), "a": float(abs(a)), "cov": None, "method": "grid"}

# --- Fine sweep ---
def _run_single_fine_sweep(ctx: dict, center_Hz, span_Hz, points, integration_time, num_averages):
    """
    Build/compile/run a single fine sweep around `center_Hz` ± span_Hz/2 relative to LO.
    """
    rel_center = float(center_Hz - ctx["center_freq"])
    start = rel_center - span_Hz/2
    stop  = rel_center + span_Hz/2

    sweep = LinearSweepParameter(
        uid=f"{ctx['qubit']}_indiv_freq_{int(center_Hz)}",
        start=start, stop=stop, count=int(points), axis_name="Frequency [Hz]"
    )
    exp = _build_experiment(
        ctx,
        sweep,
        uid=f"Resonator Spectroscopy (individual_spec @ {center_Hz/1e9:.6f} GHz)",
        integration_time=integration_time,
        num_averages=num_averages,
    )
    exp.set_calibration(_make_calibration(ctx, sweep))
    exp.set_signal_map(_make_signal_map(ctx))

    compiled = ctx["session"].compile(exp)
    result = ctx["session"].run(compiled)

    amps = np.asarray(result.acquired_results["res_spec"].data).ravel()
    frel = np.asarray(result.acquired_results["res_spec"].axis[0]).ravel()
    fabs = frel.astype(float) + float(ctx["center_freq"])

    amplitude = np.abs(amps)
    power_W   = (amplitude**2)/50.0
    y_dbm     = 10.0*np.log10(np.maximum(power_W, 1e-20)/1e-3)
    y_mag     = amplitude

    return {"center_Hz": center_Hz, "span_Hz": span_Hz, "points": points,
            "result": result, "freq_Hz": fabs, "amp_complex": amps,
            "amp_dBm": y_dbm, "amp_mag": y_mag}

def individual_spec_run_and_fit(
    ctx: dict,
    *,
    fine_span_Hz: float = 2e6,
    fine_points: int = 801,
    fine_integration_time: float | None = None,
    fine_num_averages: int | None = None,
    targets_Hz: np.ndarray | list | None = None,
):
    """
    (individual_spec) For each target frequency, do a fine sweep and fit |S21| magnitude.
    Also records the amplitude-min frequency (dip) and the fitted fr, both in GHz.
    """
    if targets_Hz is None:
        if ctx["dips_freq"] is None:
            raise RuntimeError("No targets. Run overall_spec_analyze() or pass targets_Hz.")
        targets_Hz = ctx["dips_freq"]

    ctx["individual_runs"] = []

    # initialize arrays
    ctx["min_freqs_GHz"] = []
    ctx["fr_fit_GHz"] = []

    for f0 in np.asarray(targets_Hz, dtype=float):
        run = _run_single_fine_sweep(
            ctx,
            center_Hz=f0,
            span_Hz=float(fine_span_Hz),
            points=int(fine_points),
            integration_time=(ctx["integration_time"] if fine_integration_time is None else float(fine_integration_time)),
            num_averages=(ctx["num_averages"] if fine_num_averages is None else int(fine_num_averages)),
        )
        # Magnitude-only fit
        fit = fit_notch_magnitude(run["freq_Hz"], run["amp_mag"])

        # Qi estimate with cos(phi)
        Qi = None
        try:
            Ql  = float(fit["Ql"])
            Qcm = float(fit["Qc_mag"])
            phi = float(fit["phi"])
            denom = (1.0/Ql) - (np.cos(phi)/Qcm)
            if denom > 0 and np.isfinite(denom):
                Qi = 1.0 / denom
            else:
                Qi = None
        except Exception:
            Qi = None
        fit["Qi"] = Qi

        # record min-frequency and fitted fr
        i_min = int(np.argmin(run["amp_mag"]))
        min_freq_GHz = float(run["freq_Hz"][i_min] / 1e9)
        fr_fit_GHz   = float(fit["fr"] / 1e9)
        ctx["min_freqs_GHz"].append(min_freq_GHz)
        ctx["fr_fit_GHz"].append(fr_fit_GHz)

        run["fit"] = fit
        ctx["individual_runs"].append(run)

    ctx["min_freqs_GHz"] = np.asarray(ctx["min_freqs_GHz"], dtype=float)
    ctx["fr_fit_GHz"]    = np.asarray(ctx["fr_fit_GHz"], dtype=float)

    # Save per-dip CSV and summary (use q1..qn)
    rows = []
    for k, run in enumerate(ctx["individual_runs"], start=1):
        q_label = f"q{k}"
        run["q_label"] = q_label

        freq  = run["freq_Hz"]
        y_lin = run["amp_mag"]
        y_dbm = run["amp_dBm"]

        prefix = ctx["outdir"] / f"{ctx['results_prefix']}_res_spec_{q_label}"
        np.savetxt(
            f"{prefix}_individual_spec.csv",
            np.column_stack([freq, y_lin, y_dbm]),
            delimiter=",",
            header="freq_Hz, abs_amp(|S21|), amplitude_dBm",
        )

        ffit = run["fit"]
        rows.append([
            q_label, k, run["center_Hz"],
            ffit["fr"], ffit["Ql"], ffit["Qc_mag"], ffit["Qi"], ffit["phi"], ffit["a"], ffit["method"]
        ])

    if rows:
        np.savetxt(
            ctx["outdir"] / f"{ctx['results_prefix']}_individual_spec_fits.csv",
            np.array(rows, dtype=object), fmt="%s", delimiter=",",
            header="q_label, peak_index, center_Hz_request, fr_fit_Hz, Ql, Qc_mag, Qi, phi_rad, a_mag, method",
        )

    return ctx["individual_runs"]

def _fmt_Q(Q):
    if Q is None or not np.isfinite(Q):
        return r"$\mathrm{N/A}$"
    if Q >= 1e6:
        return f"{Q/1e6:.2f}M"
    elif Q >= 1e3:
        return f"{Q/1e3:.1f}k"
    else:
        return f"{Q:.0f}"

def individual_spec_plot(ctx: dict, save_png: bool = True):
    """
    (individual_spec) Plot |S21| in dBm + phase.
    Align model phase to data by estimating a constant offset (theta0) and a
    linear-in-frequency term (2*pi*tau*f) on off-resonant points (±2*FWHM excluded).
    """
    if not ctx["individual_runs"]:
        raise RuntimeError("Run individual_spec_run_and_fit() first.")

    from matplotlib.ticker import ScalarFormatter

    figs = []; metas = []
    for k, run in enumerate(ctx["individual_runs"], start=1):
        f     = np.asarray(run["freq_Hz"], float)      # Hz
        ymag  = np.asarray(run["amp_mag"], float)      # linear |S21|
        amps  = np.asarray(run["amp_complex"], complex)
        fit   = run["fit"]
        q_lbl = run.get("q_label", f"q{k}")

        # Model predictions from magnitude-fit params (no delay terms)
        s21_fit  = s21_notch_complex(f, fit["fr"], fit["Ql"], fit["Qc_mag"], fit["phi"], fit["a"])
        ymag_fit = np.abs(s21_fit)

        # Convert linear amplitude -> dBm (assumes 50 Ω)
        power_W_data = (ymag     ** 2) / 50.0
        power_W_fit  = (ymag_fit ** 2) / 50.0
        y_dbm_data = 10.0 * np.log10(np.maximum(power_W_data, 1e-20) / 1e-3)
        y_dbm_fit  = 10.0 * np.log10(np.maximum(power_W_fit,  1e-20) / 1e-3)

        # Unwrapped phases (raw; no detrending)
        phase_data = np.unwrap(np.angle(amps))
        phase_fit  = np.unwrap(np.angle(s21_fit))

        # Off-resonant mask (exclude ±2*FWHM)
        fr   = float(fit["fr"])
        Ql   = float(fit["Ql"])
        fwhm = fr / Ql if (np.isfinite(fr) and np.isfinite(Ql) and Ql > 0) else np.nan
        if np.isfinite(fwhm):
            band_lo = fr - 2.0 * fwhm
            band_hi = fr + 2.0 * fwhm
            mask = (f < band_lo) | (f > band_hi)
        else:
            mask = np.ones_like(f, dtype=bool)

        # Phase difference
        delta = np.unwrap(phase_data - phase_fit)

        # Robust least-squares for delta ≈ (2*pi*tau)*f + theta0 over off-resonant points
        if np.count_nonzero(mask) >= 2:
            A = np.column_stack([2.0*np.pi*f[mask], np.ones(np.count_nonzero(mask))])
            sol, *_ = np.linalg.lstsq(A, delta[mask], rcond=None)
            tau_est, theta0_est = float(sol[0]), float(sol[1])  # tau [s], theta0 [rad]
        else:
            tau_est, theta0_est = 0.0, float(np.median(delta))

        # Store alignment params into the fit dict for later inspection
        fit["tau_s"] = tau_est
        fit["phase_offset"] = theta0_est

        # Apply alignment to model phase
        phase_fit_aligned = phase_fit + (2.0*np.pi*tau_est*f + theta0_est)

        # Legend text
        Qi = fit.get("Qi")
        legend_txt = (
            rf"$f_{{r}} = {fit['fr']/1e9:.6f}\,\mathrm{{GHz}}$" + "\n" +
            rf"$Q_{{i}} = {_fmt_Q(Qi)}$" + "\n" +
            rf"$Q_{{c}} = {_fmt_Q(fit['Qc_mag'])}$" + "\n" +
            rf"$Q_{{l}} = {_fmt_Q(fit['Ql'])}$"
        )

        fGHz = f / 1e9
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(6.75, 4.5), sharex=True,
            gridspec_kw={'height_ratios': [1.5, 1.0]}
        )

        # Magnitude subplot (in dBm)
        ax1.plot(fGHz, y_dbm_data, "o", label="|S21| data", markersize=3)
        ax1.plot(fGHz, y_dbm_fit,  "--", label=legend_txt)
        ax1.set_ylabel("|S21| [dBm]")
        ax1.set_title(f"Resonator Spectroscopy — individual_spec (Q{k})")
        ax1.legend(fontsize=10, loc="best")

        # Phase subplot (aligned model)
        ax2.plot(fGHz, phase_data,         "o",  label="phase data", markersize=3)
        ax2.plot(fGHz, phase_fit_aligned,  "--", label="phase fit")
        ax2.set_xlabel("Frequency [GHz]")
        ax2.set_ylabel("Phase [rad]")
        ax2.legend(fontsize=10, loc="best")

        # ---- Disable axis offset & scientific notation on X for both subplots ----
        fmt = ScalarFormatter(useOffset=False, useMathText=False)
        fmt.set_scientific(False)
        for ax in (ax1, ax2):
            ax.xaxis.set_major_formatter(fmt)
            ax.ticklabel_format(style="plain", useOffset=False, axis="x")
        # --------------------------------------------------------------------------

        plt.tight_layout()

        png = None
        if save_png:
            png = ctx["outdir"] / f"{ctx['results_prefix']}_res_spec_{q_lbl}.png"
            fig.savefig(png, dpi=600)
        figs.append(fig); metas.append({"png": str(png) if png else None})

    return figs, metas

# ============================== #
# STAGE 3: individual_spec3d_*   #
# ============================== #
def _make_calibration_for_3d(ctx: dict, frequency_source, *, meas_amp_lin, meas_range_db, acq_amp_lin=None, acq_range_db=None):
    calib = Calibration()
    calib["measure"] = SignalCalibration(
        oscillator=Oscillator(uid="qa_osc_0", frequency=frequency_source, modulation_type=ModulationType.HARDWARE),
        local_oscillator=Oscillator(uid="qa_lo", frequency=ctx["center_freq"]),
        range=float(meas_range_db),
        amplitude=float(meas_amp_lin),
    )
    calib["acquire"] = SignalCalibration(
        range=float(ctx["acquire_range"] if acq_range_db is None else acq_range_db),
        amplitude=float(1.0 if acq_amp_lin is None else acq_amp_lin),
    )
    return calib

def individual_spec3d_run(
    ctx: dict,
    *,
    dip_index: int | None = 1,
    center_Hz: float | None = None,
    # choose ONE: span_Hz OR span_low_Hz/span_high_Hz
    span_low_Hz: float | None = 2e6,
    span_high_Hz: float | None = 2e6,
    points: int = 201,
    integration_time: float | None = None,
    num_averages: int | None = None,
    measure_amps: np.ndarray | list | None = None,   # values interpreted as ΔdB; linear scale applied internally
    base_db_offset: float = 0.0,
    edge_points_for_baseline: int = 10,
):
    if center_Hz is None:
        if not (ctx.get("dips_freq") is not None and len(ctx["dips_freq"]) > 0):
            raise RuntimeError("No dip info. Provide center_Hz or run overall_spec_analyze().")
        if dip_index is None or dip_index < 1 or dip_index > len(ctx["dips_freq"]):
            raise ValueError(f"dip_index must be in [1, {len(ctx['dips_freq'])}].")
        center_Hz = float(ctx["dips_freq"][dip_index - 1])
    center_Hz = float(center_Hz)

    span_low_Hz  = float(span_low_Hz if span_low_Hz is not None else 0.0)
    span_high_Hz = float(span_high_Hz if span_high_Hz is not None else 0.0)

    rel_center = center_Hz - float(ctx["center_freq"])
    start = rel_center + span_low_Hz
    stop  = rel_center + span_high_Hz

    int_time = float(ctx["integration_time"] if integration_time is None else integration_time)
    navg     = int(ctx["num_averages"]   if num_averages    is None else num_averages)

    if measure_amps is None:
        raise ValueError("Provide 'measure_amps' (list/array of dB steps).")
    measure_amps = np.asarray(measure_amps, dtype=float).ravel()
    if measure_amps.size == 0:
        raise ValueError("'measure_amps' is empty.")

    rows_dbnorm  = []
    y_vals_dB    = []
    freqs_abs_Hz = None

    base_meas_amp_lin  = float(ctx["measure_amp"])

    for a in measure_amps:
        amp_ratio = 10**(-a/20)

        row_meas_amp_lin  = base_meas_amp_lin  * amp_ratio
        sweep = LinearSweepParameter(
            uid=f"{ctx['qubit']}_3d_freq_{int(center_Hz)}_amp_{a:.2f}dB",
            start=start, stop=stop, count=int(points), axis_name="Frequency [Hz]"
        )

        exp = _build_experiment(
            ctx,
            sweep,
            uid=f"Res 3D @ {center_Hz/1e9:.6f} GHz (Δ={a:.2f} dB)",
            integration_time=int_time,
            num_averages=navg,
        )
        exp.set_calibration(_make_calibration_for_3d(
            ctx,
            sweep,
            meas_amp_lin=row_meas_amp_lin,
            meas_range_db=ctx["measure_range"],
            acq_range_db=ctx["acquire_range"],
        ))
        exp.set_signal_map(_make_signal_map(ctx))

        compiled = ctx["session"].compile(exp)
        result   = ctx["session"].run(compiled)

        amps_cplx = np.asarray(result.acquired_results["res_spec"].data).ravel()
        aabs = np.abs(amps_cplx)
        frel = np.asarray(result.acquired_results["res_spec"].axis[0]).ravel()
        fabs = frel.astype(float) + float(ctx["center_freq"])
        if freqs_abs_Hz is None:
            freqs_abs_Hz = fabs

        # Detrend in linear domain using edge points → compute normalized dB row
        n = aabs.size
        EPS = 1e-12
        if edge_points_for_baseline and n >= 4:
            edge_pts  = int(min(max(edge_points_for_baseline, 1), (n - 2)//2))
            left_db   = 20*np.log10(max(np.mean(aabs[:edge_pts]), EPS))
            right_db  = 20*np.log10(max(np.mean(aabs[-edge_pts:]), EPS))
            baseline_db = np.linspace(left_db, right_db, n)
            row_db = 20*np.log10(np.maximum(aabs, EPS)) - baseline_db
        else:
            row_db = 20*np.log10(np.maximum(aabs, EPS))

        rows_dbnorm.append(row_db)
        y_vals_dB.append(base_db_offset + ctx["measure_range"] + 20.0*np.log10(max(row_meas_amp_lin, 1e-12)))

    ctx["_spec3d"] = {
        "center_Hz": center_Hz,
        "start_Hz": center_Hz + span_low_Hz,
        "stop_Hz":  center_Hz + span_high_Hz,
        "freq_Hz":  freqs_abs_Hz,
        "meas_amp_dB": np.array(y_vals_dB),
        "data_db_norm": np.vstack(rows_dbnorm),
        "measure_amps": measure_amps,
        "integration_time": int_time,
        "num_averages": navg,
        "dip_index": dip_index
    }

    return ctx["_spec3d"]

def individual_spec3d_plot(
    ctx: dict,
    *,
    save_png: bool = True,
    clim: tuple[float, float] | None = None,
    show_dips: bool = False,                # (kept for API compatibility)
    use_stage1_dips: bool = True,           # (kept for API compatibility)
    dip_freqs_Hz: list | np.ndarray | None = None,  # (kept)
    dip_indices: list[int] | None = None,          # (kept)
    dip_style: dict | None = None,                 # (kept)
    y_label: str = "Sweep Amplitude [dB]",
    title: str | None = None,
):
    if not ctx.get("_spec3d"):
        raise RuntimeError("Run individual_spec3d_run() first.")

    D = ctx["_spec3d"]
    fGHz = D["freq_Hz"] / 1e9

    fig, ax = plt.subplots(figsize=(6, 4.5))
    pcm = ax.pcolormesh(
        fGHz,
        D["meas_amp_dB"],
        D["data_db_norm"],
        shading="auto",
        cmap="plasma",
    )
    ax.set_xlim(np.min(fGHz), np.max(fGHz))
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel(y_label)
    ax.set_title(title or f"Resonator 3D spec @ {D['center_Hz']/1e9:.6f} GHz")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("ΔSignal [dB]")

    if clim is not None:
        pcm.set_clim(*clim)
    if "two_dip_indices" in D and D["two_dip_indices"] is not None:
        dips_fGHz = (np.asarray(D["two_dip_freqs_Hz"], float) / 1e9)
        for x in dips_fGHz:
            ax.axvline(x, ls="--", lw=1.2, color="white", alpha=0.9)
        mid_x = float(np.mean(dips_fGHz))
        y_vals = np.asarray(D["meas_amp_dB"], float)
        mid_y = float(np.median(y_vals))
        df_MHz = D["two_dip_delta_Hz"] / 1e6
        ax.text(
            mid_x, mid_y, f"Δf = {df_MHz:.3f} MHz",
            ha="center", va="center",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="0.2", ec="0.9", alpha=0.6, pad=0.3),
            color="white",
        )
    plt.tight_layout()
    png = None
    dip_index = D.get("dip_index", "X")
    if save_png:
        png = ctx["outdir"] / f"{ctx['results_prefix']}_res_spec3d_Q{dip_index}.png"
        fig.savefig(png, dpi=600)
    return fig, {"png": str(png) if png else None}

def spec3d_find_two_vertical_dips(
    ctx: dict,
    *,
    percentile: float = 10.0,     # low percentile across rows to form a 1-D "darkness" profile
    min_sep_bins: int = 5,        # minimum frequency-bin separation between the two dips
    prominence_db: float = 0.8,   # how deep a dip must be to count (in dB on the profile)
) -> dict:
    """
    From ctx['_spec3d'], find the two most prominent *vertical* dips (vs frequency).
    Returns a dict with indices, freqs (Hz), and spacing (Hz). Stores the same in ctx['_spec3d'].
    """
    if not ctx.get("_spec3d"):
        raise RuntimeError("Run individual_spec3d_run() first.")

    from scipy.signal import find_peaks

    D = ctx["_spec3d"]
    Z = np.asarray(D["data_db_norm"], dtype=float)      # shape: (Ny, Nx)
    f = np.asarray(D["freq_Hz"], dtype=float)           # shape: (Nx,)

    profile = np.percentile(Z, percentile, axis=0).astype(float)  # length Nx

    peaks, props = find_peaks(-profile, distance=max(1, int(min_sep_bins)), prominence=prominence_db)

    if peaks.size < 2:
        order = np.argsort(profile)  # ascending (lowest first)
        chosen = []
        for idx in order:
            if not chosen or all(abs(idx - c) >= min_sep_bins for c in chosen):
                chosen.append(int(idx))
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            raise RuntimeError("Could not find two separated vertical dips.")
        dip_idx = np.array(sorted(chosen))
    else:
        best = peaks[np.argsort(profile[peaks])]
        dip_idx = [int(best[0])]
        for i in best[1:]:
            if abs(i - dip_idx[0]) >= min_sep_bins:
                dip_idx.append(int(i))
                break
        if len(dip_idx) < 2:
            for i in best[2:]:
                if all(abs(i - j) >= min_sep_bins for j in dip_idx):
                    dip_idx.append(int(i))
                    break
        if len(dip_idx) < 2:
            raise RuntimeError("Found dips but couldn't enforce min_sep_bins.")
        dip_idx = np.array(sorted(dip_idx))

    dip_freqs_Hz = f[dip_idx]
    df_Hz = float(abs(dip_freqs_Hz[1] - dip_freqs_Hz[0]))

    out = {
        "dip_indices": dip_idx.tolist(),
        "dip_freqs_Hz": dip_freqs_Hz.tolist(),
        "delta_f_Hz": df_Hz,
        "profile": profile,   # returned for debugging/inspection
    }

    # cache for plotting
    D["two_dip_indices"] = dip_idx
    D["two_dip_freqs_Hz"] = dip_freqs_Hz
    D["two_dip_delta_Hz"] = df_Hz
    D["two_dip_profile_percentile"] = float(percentile)
    return out

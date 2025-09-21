import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
import pandas as pd

from measurement import *

def run_T1(device_setup, session,
           qubit_label = 'Q1'
           num_averages = 12,
           delay_start = 1e-8
           delay_stop,
           delay_num,
           drive_length,
           drive_range,
           drive_amp,
           measure_length,
           measure_range,
           measure_amp,
           acquire_range,
           rlx_time = 150e6,
           ):
    x180 = pulse_library.const(uid="x180", length=drive_length, amplitude=drive_amp)
    readout_pulse = pulse_library.const(uid="readout_pulse", length=measure_length, amplitude=measure_amp)
    delay_sweep = LinearSweepParameter(uid="delay", start=delay_start, stop=delay_stop, count=delay_num)
    def experiment(num_averages, delay_sweep, x180, readout_pulse, measure_length, rlx_time):
        # Create Experiment
        exp = Experiment(
            uid="T1 experiment",
            signals=[
                ExperimentSignal("drive"),
                ExperimentSignal("measure"),
                ExperimentSignal("acquire"),
            ],)
        with exp.acquire_loop_rt(
            uid="shots",
            count=pow(2, num_averages),
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp.sweep(
                uid="sweep", parameter=delay_sweep):
                with exp.section(uid="qubit_excitation"):
                    exp.play(signal="drive", pulse=x180)
                    exp.delay(signal="drive", time=delay_sweep)
                with exp.section(uid="qubit_readout", play_after="qubit_excitation"):
                    exp.measure(
                            measure_signal='measure',
                            measure_pulse=readout_pulse,
                            acquire_signal="acquire",
                            integration_length = measure_length,
                            handle="t1_exp",
                            reset_delay = 1e-8,)
                with exp.section(uid="relax", length=rlx_time):
                    exp.reserve(signal="measure")
                    exp.reserve(signal = "acquire")
        return exp
    exp = experiment(num_averages, delay_sweep, x180, readout_pulse, measure_length, rlx_time)
    exp.set_signal_map(signal_map_default(device_setup))
    exp.set_calibration(calibration(qb_freq, qb_lo_freq, res_freq, res_lo_freq, drive_range, measure_range, acquire_range))
    compiled_t1 = session.compile(exp)


def T1_plot(T1_results):
    # ---------- Settings ----------
    T1_data = T1_results.get_data("t1_exp")            # complex data
    T1_data_axis = T1_results.get_axis("t1_exp")[0]    # seconds

    # Convert to μs for plotting
    x = 1e6 * np.asarray(T1_data_axis, float)          # μs
    y = np.abs(T1_data).astype(float)                  # linear magnitude (a.u.)
    phase_rad = np.angle(T1_data)

    # Clean + sort
    m = np.isfinite(x) & np.isfinite(y)
    x, y, phase_rad = x[m], y[m], phase_rad[m]
    idx = np.argsort(x)
    x, y, phase_rad = x[idx], y[idx], phase_rad[idx]

    # ---------- Fit model (in linear units) ----------
    def T1_model(t_us, A, T1_us, C):
        return A * np.exp(-t_us / T1_us) + C

    p0 = [y.max(), max(np.median(x), 1e-6), y.min()]
    bounds = ([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf])

    popt, _ = curve_fit(T1_model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    A_fit, T1_fit_us, C_fit = popt

    # Smooth curve for display
    xf = np.linspace(x.min(), x.max(), 800)
    y_fit = T1_model(xf, *popt)

    # ---------- Plot ----------
    fig, axs = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True,
                            gridspec_kw={'height_ratios': [1.5, 1.0]})

    # Top: magnitude in a.u. with fit overlay
    axs[0].plot(x, y, 'o', ms=3, label="data")
    axs[0].plot(xf, y_fit, '-r', lw=1.8, label="fit")
    axs[0].axvline(T1_fit_us, color="grey", linestyle="--", linewidth=1.5)
    axs[0].set_ylabel("Signal (a.u.)")
    axs[0].legend([rf"$T_1$ = {T1_fit_us:.3f} μs"], loc="lower right", framealpha=0.9)
    axs[0].set_title(f"T1 Relaxation {qubit_label}", fontsize=12, pad=8)

    # Bottom: phase vs time (optional)
    axs[1].plot(x, phase_rad, 'o', ms=3, color="tab:purple")
    axs[1].set_xlabel("Delay (μs)")
    axs[1].set_ylabel("Phase (rad)")

    plt.tight_layout()

    # ---------- Save figure ----------
    fig_path = os.path.join(outdir, f"T1_experiment_{qubit_label}_{timestamp}.png")
    plt.savefig(fig_path, dpi=600)
    print(f"Saved figure to {fig_path}")

    # ---------- Save CSV ----------
    df = pd.DataFrame({
        "Delay_us": x,
        "Magnitude_linear": y,
        "Fit_linear_at_x": T1_model(x, *popt),
        "Phase_rad": phase_rad,
        "A_fit": np.full_like(x, A_fit, dtype=float),
        "T1_fit_us": np.full_like(x, T1_fit_us, dtype=float),
        "C_fit": np.full_like(x, C_fit, dtype=float),
    })
    csv_path = os.path.join(outdir, f"T1_experiment_{qubit_label}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")

    plt.show()
from laboneq.simple import *
from result_saver import * 
def initialization(emulation = False):
    """Descriptor for a QCCS consisting of a single SHFQC
    """
    descriptor_shfqc = """ 
    instruments:
    SHFQC:
    - address: DEV12296
        uid: device_shfqc

    connections:
    device_shfqc:
        - iq_signal: q0/drive_line
        ports: SGCHANNELS/1/OUTPUT
        - iq_signal: q0/measure_line
        ports: [QACHANNELS/0/OUTPUT]
        - acquire_signal: q0/acquire_line
        ports: [QACHANNELS/0/INPUT]
    """
    # Define and Load our Device Setup
    device_setup = DeviceSetup.from_descriptor(
        descriptor_shfqc,
        server_host="127.0.0.1",  # ip address of the LabOne dataserver used to communicate with the instruments
        server_port="8004",  # port number of the dataserver - default is 8004
        setup_name="UCLA_SHFQC",  # setup name
    )
    # Are we emulating? or actually creating pulses?
    emulate = emulation
    # create and connect to session
    session = Session(device_setup=device_setup)
    session.connect(do_emulation=emulate)
    return device_setup, session

def signal_map_default(device_setup):
    signal_map = {
        "drive": device_setup.logical_signal_groups[f"q0"].logical_signals["drive_line"],
        "measure": device_setup.logical_signal_groups[f"q0"].logical_signals["measure_line"],
        "acquire": device_setup.logical_signal_groups[f"q0"].logical_signals["acquire_line"],}
    return signal_map

def calibration(qb_freq, qb_lo_freq, res_freq, res_lo_freq, drive_range, measure_range, acquire_range):
    exp_calibration = Calibration()
    exp_calibration["drive"] = SignalCalibration(
        oscillator = Oscillator(uid = "ch0_osc_0", frequency = qb_freq -qb_lo_freq, 
            modulation_type=ModulationType.HARDWARE
        ),
        local_oscillator = Oscillator(uid="ch0_lo", frequency = qb_lo_freq),
        range = drive_range,
    )
    exp_calibration["measure"] = SignalCalibration(
        oscillator = Oscillator(uid = "qa_osc_0", frequency = res_freq-res_lo_freq,
            modulation_type=ModulationType.HARDWARE
        ),
        local_oscillator = Oscillator(uid="qa_lo", frequency = res_lo_freq),
        range = measure_range
    )
    exp_calibration["acquire"] = SignalCalibration(
        range = acquire_range
    )
    return exp_calibration


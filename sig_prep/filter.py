from scipy import signal

def notch_filter(x, src_fs, line_hz, Q=50):
    sig_notch = x.copy()
    for freq in line_hz:
        b, a = signal.iirnotch(freq, Q=Q, fs=src_fs)
        sig_notch = signal.filtfilt(b, a, sig_notch)
    return sig_notch


def highpass(x, src_fs, lowcut, order=5):
    b, a = signal.butter(5, lowcut / (src_fs / 2), btype="high")
    sig = signal.filtfilt(b, a, x, axis=1)
    return sig


def butter_bandpass_filter(x, lowcut, highcut, src_fs, order=5):
    nyquist = 0.5 * src_fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    sig_bandpass = signal.filtfilt(b, a, x)
    return sig_bandpass

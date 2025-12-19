import numpy as np
import time
from scipy.signal import lfilter, firwin

class ECGProcessor:
    def __init__(self, fs=500):
        self.fs = fs
        self.buf_len = fs * 10       # 10 秒 buffer
        self.raw_buf = np.zeros(self.buf_len)
        self.filtered_buf = np.zeros(self.buf_len)
        self.sq_buf = np.zeros(self.buf_len)
        self.integ_buf = np.zeros(self.buf_len)
        self.idx = 0

        # FIR filters
        N_notch = 120
        f0 = 60
        bw = 3
        self.b_notch = firwin(N_notch+1, [f0-bw/2, f0+bw/2], fs=fs, pass_zero='bandstop')
        self.zi_notch = np.zeros(len(self.b_notch)-1)

        N_bp = 120
        self.b_bp = firwin(N_bp+1, [5, 40], fs=fs, pass_zero=False)
        self.zi_bp = np.zeros(len(self.b_bp)-1)

        # moving average windows
        self.ma1_len = round(0.12*fs)
        self.b_ma1 = np.ones(self.ma1_len)/self.ma1_len
        self.zi_ma1 = np.zeros(self.ma1_len-1)

        self.ma2_len = round(0.12*fs)
        self.b_ma2 = np.ones(self.ma2_len)/self.ma2_len
        self.zi_ma2 = np.zeros(self.ma2_len-1)

        # peak detection
        self.threshold_scale = 1.5
        self.threshold = 0.01
        self.refractory_samples = round(0.2*fs)
        self.last_peak_idx = -np.inf
        self.candidate_peak_val = -np.inf
        self.candidate_peak_idx = -1
        self.above_threshold = False

        self.peak_abs_times = []

        # bpm
        self.bpm = 70

    def update(self, x):
        self.idx += 1
        # shift buffers
        self.raw_buf[:-1] = self.raw_buf[1:]
        self.raw_buf[-1] = x

        # notch
        y1, self.zi_notch = lfilter(self.b_notch, 1, [x], zi=self.zi_notch)
        # bandpass
        y2, self.zi_bp = lfilter(self.b_bp, 1, y1, zi=self.zi_bp)
        filt_val = y2[0]
        self.filtered_buf[:-1] = self.filtered_buf[1:]
        self.filtered_buf[-1] = filt_val

        # moving average 1
        ma1_out, self.zi_ma1 = lfilter(self.b_ma1, 1, [filt_val], zi=self.zi_ma1)
        cleaned = filt_val - ma1_out[0]

        # squaring
        sq = cleaned ** 2
        self.sq_buf[:-1] = self.sq_buf[1:]
        self.sq_buf[-1] = sq

        # moving average 2 (integration)
        integ_val, self.zi_ma2 = lfilter(self.b_ma2, 1, [sq], zi=self.zi_ma2)
        self.integ_buf[:-1] = self.integ_buf[1:]
        self.integ_buf[-1] = integ_val[0]

        # adaptive threshold
        running_mean = np.mean(self.integ_buf)
        self.threshold = max(1e-6, 0.99*self.threshold + 0.01*self.threshold_scale*running_mean)

        # peak detection state machine
        current_idx = self.idx
        if integ_val > self.threshold:
            if not self.above_threshold:
                self.above_threshold = True
                self.candidate_peak_val = integ_val
                self.candidate_peak_idx = current_idx
            else:
                if integ_val > self.candidate_peak_val:
                    self.candidate_peak_val = integ_val
                    self.candidate_peak_idx = current_idx
        else:
            if self.above_threshold:
                if (self.candidate_peak_idx - self.last_peak_idx) > self.refractory_samples:
                    peak_global_idx = self.candidate_peak_idx
                    peak_abs_time = peak_global_idx / self.fs
                    self.peak_abs_times.append(peak_abs_time)
                    self.last_peak_idx = self.candidate_peak_idx
                self.candidate_peak_val = -np.inf
                self.candidate_peak_idx = -1
                self.above_threshold = False

        # calculate BPM (最近 8 秒)
        t_now = current_idx / self.fs
        recent_peaks = [t for t in self.peak_abs_times if t >= t_now-8]
        self.peak_abs_times = recent_peaks
        if len(recent_peaks) >= 2:
            rr = np.diff(recent_peaks)
            mean_rr = np.mean(rr)
            self.bpm = round(60 / mean_rr)
        else:
            self.bpm = 70

        # convert to speed
        speed = np.clip((self.bpm - 70)/100, 0.1, 1.2)
        return self.bpm, speed

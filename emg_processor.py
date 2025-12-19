import numpy as np
import scipy.signal as signal
import collections

class EMGProcessor:
    def __init__(self, fs=500.0, debug=False):
        self.fs = fs
        self.debug = debug

        # -------------------------------------------------
        # 1. Band-pass filter for EMG (20–150 Hz)
        # -------------------------------------------------
        self.b_bp = signal.firwin(
            numtaps=121,
            cutoff=[20.0, 150.0],
            fs=self.fs,
            pass_zero=False
        )
        self.zi_bp = signal.lfilter_zi(self.b_bp, [1.0]) * 0

        # -------------------------------------------------
        # 2. Envelope smoothing (low-pass / moving average)
        # -------------------------------------------------
        self.env_len = int(0.05 * self.fs)   # 50 ms window
        self.b_env = np.ones(self.env_len) / self.env_len
        self.zi_env = signal.lfilter_zi(self.b_env, [1.0]) * 0

        # -------------------------------------------------
        # 3. Rolling mean normalization (adaptive baseline)
        # -------------------------------------------------
        self.norm_len = int(0.5 * self.fs)   # 0.5 s window
        self.norm_buf = collections.deque(
            [0.0] * self.norm_len,
            maxlen=self.norm_len
        )

        self.threshold_scale = 4.0
        self.threshold = 1e-6

        # -------------------------------------------------
        # 4. Peak detection state machine
        # -------------------------------------------------
        self.refractory_samples = int(0.2 * self.fs)
        self.last_trigger_idx = -self.refractory_samples
        self.sample_idx = 0
        self.above_threshold = False

    def update(self, x):
        self.sample_idx += 1

        # -------------------------------------------------
        # 1. Band-pass filtering
        # -------------------------------------------------
        y_bp, self.zi_bp = signal.lfilter(
            self.b_bp, [1.0], [x], zi=self.zi_bp
        )
        emg_filt = y_bp[0]

        # -------------------------------------------------
        # 2. Full-wave rectification
        # -------------------------------------------------
        rectified = abs(emg_filt)

        # -------------------------------------------------
        # 3. Envelope extraction
        # -------------------------------------------------
        env, self.zi_env = signal.lfilter(
            self.b_env, [1.0], [rectified], zi=self.zi_env
        )
        envelope = env[0]

        # -------------------------------------------------
        # 4. Adaptive normalization
        # -------------------------------------------------
        self.norm_buf.append(envelope)
        baseline = np.mean(self.norm_buf)
        self.threshold = max(
            1e-6,
            0.9 * self.threshold + 0.1 * (self.threshold_scale * baseline)
        )

        # -------------------------------------------------
        # 5. Event detection
        # -------------------------------------------------
        trigger = False
        if envelope > self.threshold:
            if not self.above_threshold:
                self.above_threshold = True
        else:
            if self.above_threshold:
                if (self.sample_idx - self.last_trigger_idx) > self.refractory_samples:
                    trigger = True
                    self.last_trigger_idx = self.sample_idx
                    if self.debug:
                        print(f"[EMG] Trigger at sample {self.sample_idx}")
                self.above_threshold = False

        if self.debug and self.sample_idx % 100 == 0:
            print(f"Env={envelope:.5f}, Th={self.threshold:.5f}")

        return trigger

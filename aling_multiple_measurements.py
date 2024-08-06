import numpy as np
import matplotlib.pyplot as plt
from typing import List, Literal

class AlignMultipleMeasurements:
    def __init__(self, signals: List[np.ndarray]):
        self.signals = signals

    def align_signals(self):
        lengths = [len(signal) for signal in self.signals]
        min_length = min(lengths)
        max_length = max(lengths)
        shortest_signal = self.signals[lengths.index(min_length)]

        aligned_signals = []

        for signal in self.signals:
            if np.array_equal(signal, shortest_signal):
                aligned_signals.append(signal)
                continue

            # Compute cross-correlation
            acor = np.correlate(shortest_signal, signal, mode='full')
            # -lent(signal) represents the shift where the start of signal aligns with the end of shortest_signal.
            # len(shortest_signal) this represents the shift where the end of signal aligns with the start of shortest_signal.
            lag = np.arange(-len(signal) + 1, len(shortest_signal))
            acormax_idx = np.argmax(np.abs(acor))
            lagDiff = lag[acormax_idx]

            # Align the signals based on the lag difference
            # Positive lagDiff: The signal needs to be shifted to the right.
	        # Negative lagDiff: The signal needs to be shifted to the left.
            
            if lagDiff > 0:
                aligned_signal = np.pad(signal, (lagDiff, 0), 'constant', constant_values=(0,))
                aligned_signal = aligned_signal[:min_length]  # Ensure the length does not exceed the shortest signal length
            else:
                aligned_signal = np.pad(signal, (0, -lagDiff), 'constant', constant_values=(0,))
                aligned_signal = aligned_signal[-lagDiff:]  # Ensure correct alignment

            aligned_signals.append(aligned_signal)

        for i in range(len(aligned_signals)):
            pad_length = max_length - len(aligned_signals[i])
            if pad_length < 0:
                raise ValueError("Padding length cannot be negative.")
            aligned_signals[i] = np.pad(aligned_signals[i], (0, pad_length), 'constant', constant_values=(0,))

        return aligned_signals




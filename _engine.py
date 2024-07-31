from pathlib import Path
import numpy as np
from scipy.signal import welch,resample,csd
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List , Dict , Union , Tuple    

# import _MODF_LSQ_V3 as LSQ
matplotlib.use('Qt5Agg')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
# plt.rcParams['font.weight'] = 'bold'


#%% Single measurement set ups 
class SingleMeasurement:
    def __init__(self, name, fs, file_path, description=None):
        self.name = name
        self.fs = fs
        self.file_path = file_path
        self.description = description
        self.data = None
        self.tdata = None

    def set_data(self, data):
        self.data = data
        self.tdata = np.arange(0, len(data) / self.fs, 1 / self.fs)
        return self

    def load_asc(self, sk_row=38, cols=[0, 1]):
        try:
            with self.file_path.open('r') as fd:
                self.data = np.loadtxt(fd, skiprows=sk_row, usecols=cols)
        except IndexError:
            raise Exception(f'The file {self.file_path} does not have {len(cols)} columns.')

        self.tdata = np.arange(0, len(self.data) / self.fs, 1 / self.fs)
        return self 
    
    def slice_data(self, a, b):
        if a < b and a >= 0 and b <= len(self.data):
            self.data = self.data[a:b, :]
        else:
            raise ValueError("Invalid slicing range")
        return self
    
    def resample(self, nw_fs):
        self.data = resample(self.data, int(len(self.data) * nw_fs / self.fs))
        self.fs = nw_fs
        self.tdata = np.arange(0, len(self.data) / self.fs, 1 / self.fs)
        return self 
    
    def detrend(self, order: int, columns):
        for col in columns:
            if col < self.data.shape[1]:
                # Fit a polynomial of the specified order
                x = np.arange(len(self.data[:, col]))
                coeffs = np.polyfit(x, self.data[:, col], order)
                # Evaluate the polynomial at the data points
                trend = np.polyval(coeffs, x)
                # Subtract the trend from the original data
                self.data[:, col] = self.data[:, col] - trend
            else:
                raise IndexError(f"Column index {col} out of range.")
        return self
        

class FFTDomain:



    def __init__(self, measurement, NFFT):
        self.NFFT= NFFT
        self.measurement = measurement  # SingleMeasurement instance
        self.psd = None
        self.freq = None
        self.trace_values = None
        self.f_trace = None
        self.data = measurement.data
        self.fs = measurement.fs
        self.name = measurement.name

    
    def create_initial_values(self , Nm, _S, f):
        return LSQ.create_initial_values(Nm, _S, f)
    
    @staticmethod
    def nextpow2(Acc):
        N = Acc.shape[0]
        _ex = np.round(np.log2(N),0)
        Nfft = 2**(_ex+1)
        return int(Nfft)
    
    def trace(self):
        Acc = self.data
        fs = self.fs
        Nc = Acc.shape[1]  # Calculate Nc from self.data
        AN = self.nextpow2(Acc)
        PSD = np.zeros((Nc,Nc,int(AN/2)+1),dtype=np.complex128)
        freq= np.zeros((Nc,Nc,int(AN/2)+1),dtype=np.complex128)
    
        for i in range(Nc):
            for j in range(Nc):
                f, Pxy = csd(Acc[:,i], Acc[:,j], fs, nfft=AN,nperseg=2**11,noverlap = None,window='hamming')
                freq[i,j]= f
                PSD[i,j]= Pxy
        TSx = np.trace(PSD)/len(f)
        self.f_trace = f
        self.trace_values = np.abs(TSx)

    def fft(self):
        
        self.psd = np.zeros((self.NFFT//2+1, self.data.shape[1]))  # Define psd array based on NFFT and data shape
        
        for i in range(self.data.shape[1]):
            _freq, _psd = welch(self.data[:, i], self.fs, nperseg=self.NFFT)  # Perform FFT on each data column
            self.psd[:, i] = _psd  # Store FFT results in corresponding psd column
        self.freq = _freq
    
        # Compute trace and f_trace after fft
        self.trace()
        return self
    

#%% modf lsq 

class MDOF_LSQ():
    def __init__(self, fftd):
        self.f_trace = fftd.f_trace
        self.trace_values = fftd.trace_values
        self.xo = None
        self.model = None
        self.f_trace_mod = None
        self.frequencies = None
        self.Nm = None
        self._S = None

    def initialize_model(self, f, _S=-10):
        self.frequencies = f
        self.Nm = len(f)
        self._S = _S
        self.xo = LSQ.create_initial_list(self.Nm, self._S, self.frequencies)
        # Removing 0 frequency errors
        self.f_trace_mod = self.f_trace.copy()
        self.f_trace_mod[0] = 10**-48

    def fit_model(self):
        N = len(self.f_trace_mod)
        self.model = LSQ.Model_opt(self.xo, self.f_trace_mod, self.Nm, N)
        return self.model

    def likelihood(self):
        return LSQ.likelihood(self.xo, self.f_trace_mod, self.trace_values, self.Nm, len(self.f_trace_mod))

    def MDOF_LSQ2(self):
        return LSQ.MDOF_LSQ2(self.xo, self.f_trace_mod, self.trace_values, self.Nm)


#%% Multiples set ups 

class MultipleMeasurements:
    def __init__(self, fftd_objects):
        self.fftd_objects = fftd_objects

    @property
    def psds(self):
        return [fftd.psd for fftd in self.fftd_objects]

    @property
    def freqs(self):
        return [fftd.freq for fftd in self.fftd_objects]

    def plot_trace_values(self):
        plt.figure(figsize=(10, 6))
        for fftd in self.fftd_objects:
            plt.semilogy(fftd.f_trace, fftd.trace_values, label=fftd.name)
        plt.ylabel('Trace Values')
        plt.xlabel('Frequency [Hz]')
        plt.legend()
        plt.show()

    def plot_psd(self, column=0):
        plt.figure(figsize=(10, 6))
        for fftd in self.fftd_objects:
            if column < fftd.psd.shape[1]:
                plt.semilogy(fftd.freq, fftd.psd[:, column], label=fftd.name)
        plt.ylabel('PSD [dB/Hz]')
        plt.xlabel('Frequency [Hz]')
        plt.legend()
        plt.show()
        plt.plt.grid('on')
        plt.gca().yaxis.grid(True, which='minor', linestyle='--')

    def get_rms_values(self):
        rms_values = {}
        for fftd in self.fftd_objects:
            rms = np.sqrt(np.mean(np.square(fftd.parent.data), axis=0))
            rms_values[fftd.name] = rms
                
        return rms_values
    
#%% Visualizer

 
class DataVisualizer:
    def __init__(self, data_instance):
        self.data_instance = data_instance
    
    def plot_data_time_domain(self, title, xlabel, ylabel,color = 'blue',legend = ['EW','NS','Z']):
        plt.figure(figsize=(10, 6))
        plt.suptitle(title)

        for i in range(self.data_instance.data.shape[1]):
            plt.subplot(self.data_instance.data.shape[1], 1, i + 1)
            plt.plot(self.data_instance.data[:, i],color=color)
            plt.legend([legend[i]],loc='upper right')
            plt.ylabel(ylabel)

            if i == self.data_instance.data.shape[1] - 1:
                plt.xlabel(xlabel)

        
        plt.tight_layout()
        plt.show()
        
    def plot_psd(self, mode='default',color='blue',legend:Optional[list] = None):   
        colormap = Colormap()

        if mode == 'default':
            plt.figure(figsize=(10, 6))
            for i in range(self.data_instance.psd.shape[1]):
                plt.subplot(self.data_instance.psd.shape[1], 1, i + 1)
                plt.plot(self.data_instance.freq, 10 * np.log10(self.data_instance.psd[:, i]),color=color)
                plt.ylabel('PSD [dB/Hz]')
                plt.legend([legend[i]],loc='upper right')

                if i == self.data_instance.psd.shape[1] - 1:
                    plt.xlabel('Frequency [Hz]')
                plt.suptitle('PSD of ' + self.data_instance.name)
            plt.tight_layout()
            plt.show()

        if mode == 'join':
            
            plt.figure(figsize=(10, 6))
            Nplots = self.data_instance.psd.shape[1]
            for i in range(Nplots):
                plt.plot(self.data_instance.freq, 10 * np.log10(self.data_instance.psd[:, i]), label=legend[i],
                         color = colormap.get_color(i, Nplots))
            plt.ylabel('PSD [dB/Hz]')
            plt.xlabel('Frequency [Hz]')
            plt.legend()
            plt.suptitle('PSD of ' + self.data_instance.name,y = 0.95)
            plt.grid('on')
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')  # Major grid lines
            plt.minorticks_on()  # Enable minor grid lines
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')  # Minor grid lines

            plt.show()

    def plot_spectrogram(self, nfft=256, noverlap=128, cmap='viridis', legend:Optional[list] = None):
        plt.figure(figsize=(10, 6))
        for i in range(self.data_instance.data.shape[1]):
            plt.subplot(self.data_instance.data.shape[1], 1, i + 1)
            Pxx, freqs, bins, im = plt.specgram(self.data_instance.data[:, i], NFFT=nfft, Fs=self.data_instance.fs,
                                                noverlap=noverlap, cmap=cmap)

            if not legend:
                current_legend = f"Series_{i+1}"
            plt.figtext(0.98, 0.9, current_legend, fontsize='medium',
                        verticalalignment='top',
                        horizontalalignment='right',
                        transform=plt.gca().transAxes,
                        bbox=dict(boxstyle="round,pad=0.3",facecolor='white', alpha=0.6, edgecolor='none'))
            plt.colorbar(im).set_label('Intensity [dB]')
            plt.ylabel('Frequency [Hz]')
            if i == self.data_instance.data.shape[1] - 1:
                plt.xlabel('Time [sec]')

        plt.suptitle('Spectrogram of ' + self.data_instance.name)
        plt.show()


class Colormap:
    def __init__(self):
        colors = ["000000", "#FFD700"]  # Dark grey to Gold
        n_bins = 100  # Number of bins
        cmap_name = "grey_gold"
        self.cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    def get_color(self, index, total):
        """Get a specific color from the colormap based on index and total number of items."""
        return self.cm(index / (total - 1) if total > 1 else 0)


class MeasurementFactory:
    def create_measurement(self, name, fs, file_path,description):
        return SingleMeasurement(name, fs, file_path,description)

    def create_fft_domain(self, measurement, NFFT):
        return FFTDomain(measurement, NFFT)



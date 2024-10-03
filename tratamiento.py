import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
#import PyOMA as oma
import mplcursors
from scipy.fftpack import fft
import os

#Dirección para guardar los datos
direccion = r'C:\Users\Usuario\OneDrive - enpc.fr\PC\ENPC\Stage 2A\Documentos\Results\Plots'
#%%FUNCIONES
""""
Quito los 5 filas y la 1 columna donde los datos son irrelevantes para el tratamiento
"""
def file_organizer(file): 
    file_organized = file.iloc[:,1:]
    #Quito el label de cada fila
    data = file_organized.to_numpy(dtype =float)
    return data

def figure(t,x,xlim = None, xlabel = None, ylabel = None,save = None,title = None):
    if x.ndim == 1:
        fig = plt.plot(t,x)
        
    elif x.ndim ==2:
        for value in x:
            #figure(t,value)
            fig = plt.plot(t,value)
    else:
        raise(' Only 1D or 2D arrays')
    
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if title:    
        plt.title(f'{title}')
        
    plt.legend(loc='best',fontsize = 6)
    plt.grid(linestyle = '-', linewidth = 1,which="both")
    if save:
        name = save[1]
        plt.savefig(save[0]+ f'\{name}.pdf')
    cursor = mplcursors.cursor(fig, hover=True)
    plt.show()
    
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sp.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sp.signal.filtfilt(b, a, data)
    return y

def FFT_(signals):
    signals_fft = list()
    for signal in signals:
        signal_fft = fft(signal)
        signals_fft.append(signal_fft)
    return np.array(signals_fft)
    
class Treatment:
    def __init__(self, data):
        self.data = data
        self.t = data[:,0]
        self.positions = self.data[:,1:]
        #Frecuencia 
        self.fs = len(self.t)/(self.t[-1]- self.t[0])

        
    def data_cleaned(self):
        #Quito las columnas que tienen infinitos o NaN
        #Quita los sensores que no miden todo el ensayo
        mask = ~np.isnan(self.positions).any(axis=0) & ~np.isinf(self.positions).any(axis=0)
        data_c = self.positions[:, mask]
        return data_c
    
    def initial_position(self):
        x0 = self.data_cleaned()[0,1:]
        return x0
        
    def tratamiento_señal(self):
        #t = self.t
        positions_filtered = self.data_cleaned()[:,1:]
        fs = self.fs
        
        
        # Diseño del filtro
        nyq = 0.5 * fs  # Frecuencia de Nyquist
        """"
        Tengo dudas con el valor de cutoff
        """
        cutoff = 0.2 # Frecuencia de corte del filtro
        normal_cutoff = cutoff / nyq
        
        b, a = sp.signal.butter(4, normal_cutoff,analog=False,btype = 'high')
        
        # Aplicar el filtro
        xn = list()
        for i in range(positions_filtered.shape[1]):
            
            y = sp.signal.filtfilt(b, a, positions_filtered[:,i])
            #Retomar posición de referencia como origen
            z = sp.signal.detrend(y, type='constant')
            xn.append(z)
            
        return np.array(xn)
    
    def _periodogram(self, cut = False):
        fs = self.fs        
        
        if cut == True:
            [t, signals_cut] = self.cut_1()
            signals = signals_cut
        
        else:
            signals = self.tratamiento_señal()
    
    
        periodograms = list()
        for signal in signals:
            f,Pxx_den = sp.signal.welch(signal, fs = fs
                                          ,window = 'hamming', nperseg = 253, density = 'spectrum')
            periodograms.append((f,Pxx_den))
        return periodograms
    
    def FFT(self, btype = None, distance = None, prominence = None):
        t,fs = self.t,self.fs
        signals_fft = list()
        
        #Dominio de frecuencia
        f_fft = sp.fft.fftfreq(len(t), 1/fs)[:len(t)//2]
        
        #FFT para cada señal
        signals = self.tratamiento_señal()
        for signal in signals:
            signal_fft = fft(signal)[0:len(t)//2]
            signals_fft.append(signal_fft)
            
            
        signals_fft = np.array(signals_fft)
        #full_data = np.vstack([f_fft,np.abs(signals_fft)])
        #Sorting...
        signals_maxs = np.max(np.abs(signals_fft), axis = 0)
        
        
        peaks, _ = sp.signal.find_peaks(
            signals_maxs,height = max(signals_maxs)/1.2, distance = distance, prominence = prominence)
        
        
        """
        signals_maxs = signals_maxs.tolist()
        signals_maxs = sorted(signals_maxs,reverse = True)[0:2]

        index_1,index_2 = np.where(full_data == signals_maxs[0]) , np.where(full_data == signals_maxs[1])
        maximos = f_fft[index_1[1]],f_fft[index_2[1]]
        """
        
        maximos = f_fft[peaks]
        return [f_fft, signals_fft,maximos]
    
    def cut_1(self,p = 20, distance = 8, prominence = 0.28):
        signals = self.tratamiento_señal()
        t = self.t
        #Código para encontrar los máximos locales y filtrados bajo la condición de altura
        characteristic_signal = np.average(signals,axis = 0)
        y = characteristic_signal
        y_abs = np.abs(y)
        #max_ me indica los índices en los cúales se encuentran los máximos locales
        max_, _ = sp.signal.find_peaks(y_abs,distance = distance,prominence = prominence,height = max(y_abs)/p)
        t_max = t[max_]
        #Corte del intervalo
        mask_t = (t>= t_max[0]) & (t<=t_max[-1])
        t_cut = t[mask_t]
        #Resultado de corte
        signals_cut = signals[:,max_[0]:max_[-1]+1]
        return [t_cut, signals_cut]
        
        
#%%

# ======== PRE-PROCESSING =====================================================
# To open a .txt file create a variable containing the path to the file
_file = r"C:\Users\Usuario\OneDrive - enpc.fr\PC\ENPC\Stage 2A\Documentos\Datos\USW2\Séisme\All4Wall2_GM0.csv" # Path to the txt file

# open the file with pandas and create a dataframe
# N.B. whatchout for header, separator and remove time column if present
file = pd.read_csv(_file, header = 6, sep=",") 
#file = pd.read_csv(_file) 
print(file)

#%% TRATAMIENTO DE DATOS
data = file_organizer(file)
data_0 = Treatment(data)
#%%
t = data_0.t
fs = data_0.fs
#%%
#positions = data_0.positions
positions_filtered = data_0.data_cleaned()
x0 = data_0.initial_position()
#%%
signals = data_0.tratamiento_señal()
signals_FFT = data_0.FFT()
#%% VISUALIZACIÓN
figure(t,signals[0:4],xlabel =r'$t\,[s]$', ylabel = 'Position')
#%%
#%% Periodograms
periodograms = data_0._periodogram()
for periodogram in periodograms:
    f, Pxx_den = periodogram
    fig = plt.plot(f,Pxx_den)
    plt.xlim(0, 3)
    plt.xlabel(r'$f\,[Hz]$')
    plt.ylabel(r'$P_{xx}$')
    plt.grid(linestyle = '-', linewidth = 1,which="both")
    #plt.yscale('log')
plt.savefig(direccion + '\periodogram.pdf')    
cursor = mplcursors.cursor(hover=True)
plt.show()
#%% FFT des signals
figure(signals_FFT[0],signals_FFT[1], xlim = [0,3],xlabel = r'$f\,[Hz]$')
#%%

"""
Cosas que tengo por hacer:
    1. Reducir el intervalo de tiempo
        (Para cada posición en el espacio)
        - Tocaría tomar el valor absoluto de las posiciones [x]
        - Tomo el máximo encontrado y luego corto
        - 
"""

# ========WORK-IN-PROGRESS=====================================================
#%% Ensayo para encontrar frecuencias propias
peaks, _ = sp.signal.find_peaks(
    np.abs(signals_FFT[1][0]),height = max(np.abs(signals_FFT[1][0]))/10)
frequencies = signals_FFT[0][peaks]
mask = frequencies > 0
frequencies = frequencies[mask]
frequencies = sorted(frequencies,reverse = True)
print("Frecuencias propias:", frequencies)
#%%Cortar un intervalo (Método 1)
# Lo que podría mejorara acá sería usar una señal característica
#¿Cuál sería la señal característica?: Promedio de las señales
characteristic_signal = np.average(signals,axis = 0)
y = characteristic_signal
average = np.average(y)
y_abs = np.abs(y)
max_, _ = sp.signal.find_peaks(y_abs,height = max(y_abs)/10)
#max_ son los índices donde se encuentran los picos
#max_ = sorted(max_)
t_max = t[max_]
y_max = y[max_]
print(f'{y_abs}')
#%% 
mask_t = (t>= t_max[0]) & (t<=t_max[-1])
t_cut = t[mask_t]
print(t_cut[0],t_cut[-1])
#%%
y_cut = y[max_[0]:max_[-1]+1]
print(len(y_cut))
#%%Intentaremos cortas las señales 
signals_cut = signals[:,max_[0]:max_[-1]+1]
figure(t_cut,signals_cut)
#%%
figure(t_cut,y_cut)
#%%Cortar un intervalo (Método 2)
y_b,y_f = y[:max_[0]],y[max_[0]:max_[-1]:]

p_y_b,p_y_f =np.average(y_b), np.average(y_f)
d_y_b,d_y_f =np.std(y_b), np.std(y_f)
#Encontrar el índice en el cual que super el intervalo
#%%
mask_y_b = np.abs(y_b) >= np.abs(p_y_b) + d_y_b
figure(np.linspace(1,100,len(y_b[mask_y_b])),y_b[mask_y_b])
#%%
figure(t[:max_[0]],y_b)

#%% Sauvergarder les signals
to_save = np.column_stack((t_cut,y_cut))

np.savetxt(direccion +r'\cut.csv',
           to_save, delimiter=',')
#%%

np.savetxt(direccion +r'\signals_cut.csv',
           signals_cut, delimiter=',')
#%%

np.savetxt(direccion +r'\signals.csv',
           signals, delimiter=',')
#%% Sauvergarder les signals
to_save = list()
to_save.append(np.transpose(t))
for signal in signals:
    to_save.append(signal)

np.savetxt(direccion +r'\signals.csv',
           np.array(np.transpose(to_save)), delimiter=',')
#%%
""""
Ensayos para comparar los filtros...
"""
#%% POSITIONS INITIALS
i = 20
x = positions_filtered[:,i]
x -= x[0]
x0 = np.array([signals[i],x])

figure(t,x0)


#%% Comparaison avec differents filtres

filtered_signals = sp.signal.savgol_filter(signals, window_length=53, polyorder=20, axis=0)
j = 20
unfiltered = positions_filtered[:,j]-positions_filtered[0,j]
x1 = np.array([unfiltered,signals[j],filtered_signals[j]])
figure(t,x1)


#%%
"""
Hacer el corte de señal como el profesor quiere
"""

[fig, t_cut, signals_cut] = data_0.cut_1()

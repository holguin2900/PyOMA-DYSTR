import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
#import PyOMA as oma
import mplcursors
from scipy.fftpack import fft
import os


#%%FUNCIONES

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


def FFT_(signals):
    signals_fft = list()
    for signal in signals:
        signal_fft = fft(signal)
        signals_fft.append(signal_fft)
    return np.array(signals_fft)


#%%    

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
        x0 = self.data_cleaned()[0,:]
        return x0

    def tratamiento_señal(self):
        #t = self.t
        positions_filtered = self.data_cleaned()
        fs = self.fs


        # Diseño del filtro
        nyq = 0.5 * fs  # Frecuencia de Nyquist
        """"
        Tengo dudas con el valor de cutoff
        """
        cutoff = 0.1 # Frecuencia de corte del filtro
        normal_cutoff = cutoff / nyq

        b, a = sp.signal.butter(2, normal_cutoff,analog=False,btype = 'high')

        # Aplicar el filtro
        xn = list()
        for i in range(positions_filtered.shape[1]):
            #y = positions_filtered[:,i]
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
            f,Pxx_den = sp.signal.periodogram(signal, fs = fs
                                          ,window = 'flattop', scaling='density')
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
            signals_maxs,height = max(signals_maxs)/3, distance = distance, prominence = prominence)


        """
        signals_maxs = signals_maxs.tolist()
        signals_maxs = sorted(signals_maxs,reverse = True)[0:2]

        index_1,index_2 = np.where(full_data == signals_maxs[0]) , np.where(full_data == signals_maxs[1])
        maximos = f_fft[index_1[1]],f_fft[index_2[1]]
        """

        maximos = f_fft[peaks]
        return [f_fft, signals_fft,maximos]

    def cut_1(self,p = 200, distance = 8, prominence = 0.28):
        #signals = self.tratamiento_señal()
        signals = self.data_cleaned()[:,:]
        signals -= self.initial_position()
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
    
    
    def average(self, sigma = 2.5):
        
        positions_filtered = self.tratamiento_señal()
        
        # Aplicar el filtro
        xn = list()
        for i in range(positions_filtered.shape[0]):
            #y = positions_filtered[:,i]
            y = sp.ndimage.gaussian_filter1d(positions_filtered[i], sigma = sigma)
            #Retomar posición de referencia como origen
            z = sp.signal.detrend(y, type='constant')
            xn.append(z)
            
        xn = np.array(xn)
        # Tomar un promedio
         
        x = xn[0::3]
        y = xn[1::3]
        z = xn[2::3]
        
        x_p,y_p,z_p = np.average(x,axis = 0),np.average(y,axis = 0),np.average(z,axis = 0)
        
        return [x_p,y_p,z_p]
    
    
    def average_periodogram(self, cut = False):
        fs = self.fs

        if cut == True:
            [t, signals_cut] = self.cut_1()
            signals = signals_cut

        else:
            directions = self.average()


        periodograms = list()
        for direction in directions:
            f,Pxx_den = sp.signal.periodogram(direction, fs = fs
                                          ,window = 'flattop', scaling='density')
            periodograms.append((f,Pxx_den))
        return periodograms
    
    
#%%
"""
Automatización para carpetas
"""
#%%
""""
Programar para todo una carpeta
"""
# Ruta del directorio de entrada y salida
#input_dir = str(input())
input_dir = r'C:\Users\Usuario\OneDrive - enpc.fr\PC\ENPC\Stage 2A\Documentos\Datos\USW2\Séisme'
input_dir = rf'{input_dir}'
output_dir = input_dir + '\Results'
output_dir = rf'{output_dir}' 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f'Carpeta creada en: {output_dir}')
else:
    print(f'La carpeta ya existe en: {output_dir}')
#%%
data_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".csv")]
data_files = sorted(data_files)
names = list()
for file in data_files:
    #name = os.path.basename(file)
    name = os.path.splitext(os.path.basename(file))[0]
    names.append(name)
#%%
FFT_folder = list()
signals_folder = list()
periodograms_folder = list()


#%% Signals vs t
folder = output_dir+'\Signals vs t'
if not os.path.exists(folder):
    os.makedirs(folder)
for file,name in zip(data_files,names):
    #Uso de pandas
    _file = pd.read_csv(rf'{file}', header = 6, sep=",")
    data = file_organizer(_file)
    signals_folder.append(data)
    data_0 = Treatment(data)  
    #Signals
    signals = data_0.tratamiento_señal()
    t = data_0.t
    fs = data_0.fs
    
    
    figure(t,signals,
           xlabel =r'$t\,[s]$', ylabel = 'Position' , save = [folder,'signals_'+name], title = name)
    
#%% Periodograms
folder = output_dir+'\Periodograms'
if not os.path.exists(folder):
    os.makedirs(folder)
    
    
for file,name in zip(data_files,names):
    #Uso de pandas
    _file = pd.read_csv(rf'{file}', header = 6, sep=",")
    data = file_organizer(_file)
    data_0 = Treatment(data) 
    periodograms = data_0._periodogram(cut = False)
    #periodograms_folder.append(periodograms)
    for periodogram in periodograms:
        f, Pxx_den = periodogram
        
        
        fig = plt.plot(f,Pxx_den)
        
    plt.xlim(0, 10)
    plt.ylim([1e-8, 1e3])
    plt.semilogy()
    plt.xlabel(r'$f\,[Hz]$')
    #plt.ylabel(r'$PSD \,[cm^2/Hz]$')
    plt.ylabel(r'$PSD \,[mm^2]$')
    plt.grid(linestyle = '-', linewidth = 1,which="both")
    plt.title(f'{name}')
    plt.savefig(folder + f'\periodogram_{name}.pdf')
    plt.show()

#%% FFT
folder = output_dir+'\FFT'
if not os.path.exists(folder):
    os.makedirs(folder)
    
    
for file,name in zip(data_files,names):
    #Uso de pandas
    _file = pd.read_csv(rf'{file}', header = 6, sep=",")
    data = file_organizer(_file)
    #data_folder.append(data)
    data_0 = Treatment(data)  
    signals_FFT = data_0.FFT(distance = 1e2, prominence = 2e2)
    FFT_folder.append(signals_FFT)
    
    figure(
        signals_FFT[0],np.abs(signals_FFT[1]), xlim = [0,10],
        ylabel = r'$|\hat{X}(f)|$',xlabel = r'$f\,[Hz]$',save = [folder,'FFT_'+name]
        ,title =f'{name}\n' r'$f =$'+ f'{np.around(signals_FFT[2],2).tolist()} ' +r'$[Hz]$')
    
    print(signals_FFT[2])
    np.savetxt(folder +fr'\frequencies_{name}.txt',
               signals_FFT[2], delimiter=',')

#%% PyOMA_cut
folder = output_dir+'\Signals Cut'
if not os.path.exists(folder):
    os.makedirs(folder)
    
    
for file,name in zip(data_files,names):
    #Uso de pandas
    _file = pd.read_csv(rf'{file}', header = 6, sep=",")
    data = file_organizer(_file)
    data_0 = Treatment(data)
    [t_cut, signals_cut] = data_0.cut_1()#p = 2e2, distance = 1e2, prominence= 1e-1)
    
    figure(t_cut,signals_cut,
           xlabel= r'$t\,[s]$',ylabel= r'Position', save = [folder,'signal_cut_'+name], title = name)
    
    
    np.savetxt(folder +fr'\signals_cut_{name}.csv',
               signals_cut, delimiter=',')
    
#%% 3D raw input
folder = output_dir+r'\raw 3D'
if not os.path.exists(folder):
    os.makedirs(folder)

for file,name in zip(data_files,names):
    #Uso de pandas
    _file = pd.read_csv(rf'{file}', header = 6, sep=",")
    data = file_organizer(_file)
    data_0 = Treatment(data)
    
    r0 = data_0.initial_position()
    x0 = r0[::3]
    y0 = r0[1::3]
    z0 = r0[2::3]
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")

    ax.scatter3D(x0, y0, z0, color = "green")
    ax.set_xlabel('X-axis', fontweight ='bold')
    ax.set_ylabel('Y-axis', fontweight ='bold')
    ax.set_zlabel('Z-axis', fontweight ='bold')
    plt.title(f'{name}\n sensors = {len(x0)} Initial positions')
    
    plt.savefig(
        rf'{folder}\{name}_input3D.pdf')
    
    plt.show()
    
#%% Filtro y promedio
folder = output_dir+'\Signals_average vs t'
if not os.path.exists(folder):
    os.makedirs(folder)
label_lst = ['x','y','z']


for file,name in zip(data_files,names):
    #Uso de pandas
    _file = pd.read_csv(rf'{file}', header = 6, sep=",")
    data = file_organizer(_file)
    data_0 = Treatment(data) 
    data_filtered, t = data_0.average(),data_0.t    
    #Plot
    for i in range(len(data_filtered)):
        plt.plot(t, data_filtered[i], label = f'direction: {label_lst[i]}')
        
    plt.legend(loc='best',fontsize = 6)
    plt.grid(linestyle = '-', linewidth = 1,which="both")
    plt.ylabel(r'$Position\, [mm]$')
    plt.xlabel(r'$t\, [s]$')
    plt.title(f'{name}')
    plt.savefig(
        rf'{folder}\{name}_average_vs_t.pdf')
    plt.show()
#%% Densidad espectal promediada
folder = output_dir+'\Periodograms Average'
if not os.path.exists(folder):
    os.makedirs(folder)

label_lst = ['x','y','z']
for file,name in zip(data_files,names):
    #Uso de pandas
    _file = pd.read_csv(rf'{file}', header = 6, sep=",")
    data = file_organizer(_file)
    data_0 = Treatment(data) 
    periodograms = data_0.average_periodogram(cut = False)
    #periodograms_folder.append(periodograms)
    for periodogram,label in zip(periodograms,label_lst):
        f, Pxx_den = periodogram
        
        
        fig = plt.plot(f,Pxx_den, label = f'direction: {label}')
        
    plt.xlim(0, 10)
    plt.ylim([1e-8, 1e3])
    plt.semilogy()
    plt.xlabel(r'$f\,[Hz]$')
    #plt.ylabel(r'$PSD \,[mm^2/Hz]$')
    plt.ylabel(r'$PSD \,[mm^2]$')
    plt.legend(loc='best',fontsize = 6)
    plt.grid(linestyle = '-', linewidth = 1,which="both")
    plt.title(f'{name}')
    plt.savefig(folder + f'\{name}_periodogram_average.pdf')
    plt.show()
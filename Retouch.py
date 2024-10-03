import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
import PyOMA as oma
import mplcursors
from scipy.fftpack import fft as fft
import os
import json
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
        signals = self.data_cleaned()[:,1:]
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



#%%
def frequency_finder(data):
  frequencies = data['Reduced Poles']['Frequency'].to_numpy()

  histogram = np.histogram(frequencies, bins = 500)
  frecuencia_hist = histogram[0]
  index = np.argmax(frecuencia_hist)

  FreQ = histogram[1][index]

  return FreQ



"""
Esta función se encargará de extraer los resultados solicitados (e.g., frecuencia, amortiguamiento, y modos normales)
de cada sensor para así que sea más fácil la implementación del resto de cosas
"""



def oma_Time(_input, FreQ = None,fs = 1e2, br = 28, ordmax = None, lim = (5e-2, 5e-1,5e-2,1e-1)):

  #SSIdat_ = oma.SSIcovStaDiag(_input, fs,br, ordmax = ordmax,lim=(5e-2, 5e-2,0.02,1e-1), method =['1'])
  SSIdat_ = oma.SSIcovStaDiag(_input, fs,br, ordmax = ordmax,lim=(5e-2, 5e-1,5e-2,1e-1), method =['1'])



  if not FreQ:
    FreQ = frequency_finder(SSIdat_[1])


  deltaf = 0.2
  while True:
    try:
      Res_SSIcov = oma.SSIModEX(
          [FreQ], SSIdat_[1], aMaClim=0.85,deltaf = deltaf)
      break

    except Exception as e:
      incremento = 0.02
      print(f'Error (Agregando {incremento} a deltaf = {deltaf}):\n{e}')
      deltaf += incremento

      if deltaf >= 0.35:
        print(f'Se ha pasado el límite de deltaf = {deltaf}')
        break

    #Res_SSIcov = oma.SSIModEX(FreQ, SSIdat_[1],aMaClim=0.95,deltaf = 0.2)

  frequency, damping, mode_shape = Res_SSIcov['Frequencies'][0], Res_SSIcov['Damping'][0],Res_SSIcov['Mode Shapes']

#CAMBIAR PLOT
  ax = SSIdat_[0].gca()  # get current axes
  # Cambiar los límites del eje x
  x_lim = 3
  dx = 0.1
  x_ticks = np.arange(0, x_lim + dx, dx)
  ax.set_xticks(x_ticks)
  ax.set_xlim(2, x_lim)
  #ax.set_ylim(0, ordmax)
  ax.grid(True)
  ax.grid(linestyle = '-', linewidth = 1,which="both")
 
  plt.show()
  return ((frequency, damping, mode_shape),
          SSIdat_[0])



def oma_Frequency(FreQ,_input,fs = 1e2,df = 5e-2,npmax = 9):
  FDDsvp = oma.FDDsvp(_input, fs, df = df, pov = 0.5,window = 'blackman')

  Res_FSDD = oma.EFDDmodEX(FreQ, FDDsvp[1], sppk = 0, npmax = npmax, plot=True, method='FSDD', MAClim=0.85)

  frequency, damping, mode_shapes = Res_FSDD['Frequencies'][0][0],Res_FSDD['Damping'][0][0], Res_FSDD['Mode Shapes'][:,0]
  return frequency, damping, mode_shapes

#%%
""""
Programar para todo una carpeta
"""
# Ruta del directorio de entrada y salida
#input_dir = str(input())
input_dir = r'C:\Users\Usuario\OneDrive - enpc.fr\PC\ENPC\Stage 2A\Documentos\Datos\USW2\Bruit'
input_dir = rf'{input_dir}'
output_dir = input_dir + r'\Results\Tables'
output_dir = rf'{output_dir}'

    
    #%%
data_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".csv")]
data_files = sorted(data_files)

results_files = [os.path.join(output_dir, file) for file in os.listdir(output_dir) if file.endswith(".csv")]
results_files = sorted(results_files)


start,end = 0,19

data_files_c,results_files_c = data_files[start:end],results_files[start:end]

names = list()

for file,archivo in zip(data_files_c,results_files_c):
    #name = os.path.basename(file)
    name = os.path.splitext(os.path.basename(file))[0]
    names.append(name)

    _file = pd.read_csv(file, header = 6, sep = ",")
    _archivo = pd.read_csv(archivo).iloc[:,1:]
    
    #Filtro por arriba
    frecuencia_corte = 5
    _archivo = _archivo[_archivo['f_T'] <= frecuencia_corte]
    #Filtro de copias de sensores
    _archivo = _archivo.drop_duplicates(subset='sensores', keep='first')
    
    FreQ = sp.stats.trim_mean(_archivo['f_T'],0.1)

    #PREPROCESAMIENTO
    _data = file_organizer(_file)
    
    
    
    data = Treatment(_data)    
    fs = data.fs
    initial_position = data.initial_position()
    #Tratada
    signals_filtered = data.tratamiento_señal()
    _input = signals_filtered.transpose()[:,:]
    
    n_sensores = _input.shape[1]//3
    
    sensores_medidos = _archivo['sensores'].to_numpy()
    
    print(sensores_medidos)
    for sensor in range(n_sensores):
        if sensor not in sensores_medidos:
            print(sensor)
            input = _input[:,3*sensor:3*sensor+3]
            results_T = dict()
            
         
            try:
              print(f'sensor: {name}')
              result = oma_Time(input, fs = fs, br = 28, FreQ = FreQ)
              frecuencia, amortiguamiento,modo_normal = result[0]
              modo_normal  = np.real(modo_normal)

              resultado = oma_Frequency(FreQ = [FreQ],_input = input, fs = fs, npmax= 6)
              frequency, damping, mode_shapes = resultado
              mode_shapes = np.real(mode_shapes)

              results_T['x0'] = input[0][0]
              results_T['y0']= input[0][1]
              results_T['z0'] = input[0][2]
             
              results_T['f_T'] = frecuencia
              results_T['d_T'] = amortiguamiento

              results_T['f_F'] = frequency
              results_T['d_F'] = damping

              results_T['x_T'] = modo_normal[0][0]
              results_T['y_T'] = modo_normal[1][0]
              results_T['z_T'] = modo_normal[2][0]

              results_T['x_F'] = mode_shapes[0]
              results_T['y_F'] = mode_shapes[1]
              results_T['z_F'] = mode_shapes[2]
              
              
              results_T['sensores'] = sensor
              
              print(f'Análisis modal calculado para sensor {sensor}')
              
              nueva_fila = pd.DataFrame([results_T])
              _archivo = pd.concat([_archivo,nueva_fila], ignore_index=True)
              
            except Exception as e:
              print(f'error en sensor {sensor}: {e}')
              pass
             
            
            _archivo = _archivo.sort_index()
            _archivo = _archivo.drop_duplicates(subset='sensores', keep='first')
            _archivo.to_csv(f'{output_dir}/{name}_results.csv')
            #print(sensor)
    #print(sensores_medidos)
    

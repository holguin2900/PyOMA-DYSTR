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

#%%
def frequency_finder(data):
  frequencies = data['Reduced Poles']['Frequency'].to_numpy()

  histogram = np.histogram(frequencies, bins = 500)
  frecuencia_hist = histogram[0]
  index = np.argmax(frecuencia_hist)

  FreQ = histogram[1][index]

  return [FreQ]



"""
Esta función se encargará de extraer los resultados solicitados (e.g., frecuencia, amortiguamiento, y modos normales)
de cada sensor para así que sea más fácil la implementación del resto de cosas
"""



def oma_Time(_input, FreQ = None,fs = 1e2, br = 28, ordmax = None, lim = (5e-2, 5e-1,5e-2,1e-1)):

  #SSIdat_ = oma.SSIcovStaDiag(_input, fs,br, ordmax = ordmax,lim=(5e-2, 5e-2,0.02,1e-1), method =['1'])
  SSIdat_ = oma.SSIcovStaDiag(_input, fs,br, ordmax = ordmax,lim= lim, method =['1'])


  if not FreQ:
    FreQ = frequency_finder(SSIdat_[1])
    
  deltaf = 0.02
  while True:
    try:
      Res_SSIcov = oma.SSIModEX(
          FreQ, SSIdat_[1], aMaClim=0.85, deltaf = deltaf)
      break

    except Exception as e:
      incremento = 0.02
      print(f'Error (Agregando {incremento} a deltaf = {deltaf}):\n{e}')
      deltaf += incremento

      if deltaf >= 1.5:
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



def oma_Frequency(FreQ,_input,fs = 1e2,df = 5e-2):
  FDDsvp = oma.FDDsvp(_input, fs, df = df, pov = 0.5,window = 'blackman')

  Res_FSDD = oma.EFDDmodEX(FreQ, FDDsvp[1], sppk = 0, npmax = 9, plot=True, method='FSDD', MAClim=0.85)

  frequency, damping, mode_shapes = Res_FSDD['Frequencies'][0][0],Res_FSDD['Damping'][0][0], Res_FSDD['Mode Shapes'][:,0]
  return frequency, damping, mode_shapes
#%%
def CreateCSV(_input,fs,name,outputdir,FreQ = None):
  results_T = dict(
                 sensores = list(),
                 x0 = list(),
                 y0 = list(),
                 z0 = list(),
                 f_T = list(),
                 f_F = list(),
                 d_T = list(),
                 d_F = list(),
                 x_T = list(),
                 x_F = list(),
                 y_T = list(),
                 y_F = list(),
                 z_T = list(),
                 z_F = list()
                 )
  
  for i in range(0,_input.shape[1],3):
    input = _input[:,i:i+3]
    try:

      result = oma_Time(input, fs = fs, br = 28, FreQ = FreQ)
      frecuencia, amortiguamiento,modo_normal = result[0]
      modo_normal  = np.real(modo_normal)

      resultado = oma_Frequency(FreQ = [frecuencia],_input = input, fs = fs)
      frequency, damping, mode_shapes = resultado
      mode_shapes = np.real(mode_shapes)

      #results_T['x0'].append(input[0][0])
      #results_T['y0'].append(input[0][1])
      #results_T['z0'].append(input[0][2])
     
      results_T['f_T'].append(frecuencia)
      results_T['d_T'].append(amortiguamiento)

      results_T['f_F'].append(frequency)
      results_T['d_F'].append(damping)

      results_T['x_T'].append(modo_normal[0][0])
      results_T['y_T'].append(modo_normal[1][0])
      results_T['z_T'].append(modo_normal[2][0])

      results_T['x_F'].append(mode_shapes[0])
      results_T['y_F'].append(mode_shapes[1])
      results_T['z_F'].append(mode_shapes[2])
      
      
      results_T['sensores'].append(i//3)
      
      print(f'Análisis modal calculado para sensor {i//3}')

    except Exception as e:
      print(f'error en sensor {i//3}: {e}')
      pass

  for key in results_T.keys():
    results_T[key] = np.array(results_T[key])

  tabla = pd.DataFrame.from_dict(results_T)
  tabla.to_csv(f'{outputdir}/{name}_results.csv')
  return tabla
#%%
""""
Programar para todo una carpeta
"""
# Ruta del directorio de entrada y salida
#input_dir = str(input())
input_dir = r'C:\Users\Usuario\OneDrive - enpc.fr\PC\ENPC\Stage 2A\Documentos\Datos\USW1\Séisme\Results\Signals Cut'
input_dir = rf'{input_dir}'
output_dir = input_dir + r'\Results\Tables'
output_dir = rf'{output_dir}'

if not os.path.exists(input_dir):
    os.makedirs(input_dir)
    print(f'Carpeta creada en: {input_dir}')
else:
    print(f'La carpeta ya existe en: {input_dir}')


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f'Carpeta creada en: {output_dir}')
else:
    print(f'La carpeta ya existe en: {output_dir}')
    #%%
data_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".csv")]
data_files = sorted(data_files)
names = list()
for file in data_files[:]:
    #name = os.path.basename(file)
    name = os.path.splitext(os.path.basename(file))[0]
    names.append(name)

    _file = pd.read_csv(file, sep = ",")
    _data = _file.to_numpy(dtype =float)
    
    fs = 1e2
    
    _input = _data
    print(f'Archivo: {name}')
    

    _archivo = CreateCSV(_input, fs,name,output_dir)
    
   
    
#%%

"""
    #PREPROCESAMIENTO
    _data = file_organizer(_file)
    data = Treatment(_data)


    t = data.t
    fs = data.fs
    signals = data.data_cleaned()
    initial_position = data.initial_position()
    signals -= initial_position
    #Tratada
    signals_filtered = data.tratamiento_señal()

    _input = signals_filtered.transpose()[:,:]
    print(f'Archivo: {name}')
    _archivo = CreateCSV(_input, fs,name,output_dir)
  
 #%%
 
 
 
 
 
 
data_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".csv")]
data_files = sorted(data_files)

file = data_files[-1]
_file = pd.read_csv(file, header = 6, sep = ",")


#PREPROCESAMIENTO
_data = file_organizer(_file)
data = Treatment(_data)


t = data.t
fs = data.fs
signals = data.data_cleaned()
initial_position = data.initial_position()
signals -= initial_position
#Tratada
signals_filtered = data.tratamiento_señal()
_input = signals_filtered.transpose()[:,:]
print(_input.shape)
#%%
print(_input.shape[1])
i = 267
input = _input[:,i:i+3]
#%%
SSIdat_ = oma.SSIcovStaDiag(input, fs,br = 28, ordmax = None,lim=(5e-2, 5e-1,5e-2,1e-1), method =['1'])
FreQ = frequency_finder(SSIdat_[1])



#%%


#CAMBIAR PLOT
ax = SSIdat_[0].gca()  # get current axes
# Cambiar los límites del eje x
x_lim = 5
dx = 0.2
x_ticks = np.arange(0, x_lim + dx, dx)
ax.set_xticks(x_ticks)
ax.set_xlim(0, x_lim)
#ax.set_ylim(0, ordmax)
ax.grid(True)
ax.grid(linestyle = '-', linewidth = 1,which="both")

plt.show(SSIdat_[0])

#%%
SSIdat_[0]



#%%
deltaf = 0.2
while True:
  try:
    Res_SSIcov = oma.SSIModEX(FreQ, SSIdat_[1], aMaClim=0.95,deltaf = deltaf
                          )
    print(f'deltaf = {deltaf}\n{Res_SSIcov}')
    break
  except Exception as e:
    incremento = 0.02
    print(f'Error en el sensor {i} (Agregando {incremento} a deltaf = {deltaf}):\n{e}')
    deltaf += incremento
  if deltaf >= 0.35:
    print(f'Se ha pasado el límite de deltaf = {deltaf}')
    break
print(Res_SSIcov)
#%%
FDDsvp = oma.FDDsvp(input, fs, df = 5e-2, pov = 0.5)#, window = 'blackman')
print(FDDsvp[1].keys())
#%%
Res_FSDD = oma.EFDDmodEX(FreQ, FDDsvp[1], sppk = 1,npmax = 9, plot=True, method='FSDD', MAClim=0.85)
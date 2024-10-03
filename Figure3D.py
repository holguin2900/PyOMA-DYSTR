import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
import PyOMA as oma
import mplcursors
from scipy.fftpack import fft
import os
import json
from sigfig import round
#%%

def Figure3D(modos_archivo, _archivo, c = 100 ,name = None, title = None):

  #_archivo = r'/content/All4Wall2_GM0_results.csv'
  archivo = pd.read_csv(_archivo, header = 0, sep=",", index_col = 0)
      
  with open(f'{modos_archivo}', 'r') as modos_propios:
      modos = json.load(modos_propios)
   

  x0 = np.array(modos['x0'])
  y0 = np.array(modos['y0'])
  z0 = np.array(modos['z0'])

 
  
  cut = 0
  frequency = sp.stats.trim_mean(archivo['f_T'], cut)
  dfrequency = sp.stats.mstats.trimmed_std(archivo['f_T'], limits=(cut,cut))
  frequency = round(frequency,dfrequency)
  
  damping =  sp.stats.trim_mean(archivo['d_T'], cut)
  ddamping = sp.stats.mstats.trimmed_std(archivo['d_T'], limits=(cut, cut))
  damping = round(100*damping,100*ddamping)

  
  N_x0 = c*np.array(modos['x_F'])
  N_y0 = c*np.array(modos['y_F'])
  N_z0 = c*np.array(modos['z_F'])
  
  F, D = round(modos['f'],4), round(100*modos['d'],4)
  
  f = sp.stats.trim_mean(archivo['f_F'],0.1)
  df = sp.stats.mstats.trimmed_std(archivo['f_F'], limits=(0.1, 0.1))

  f = round(f,df)   


  d =  sp.stats.trim_mean(archivo['d_F'],0.1)
  dd = sp.stats.mstats.trimmed_std(archivo['d_F'], limits=(0.1, 0.1))

  d = round(100*d,100*dd)

  fig = plt.figure(figsize = (16, 9))
  ax = plt.axes(projection ="3d", proj_type = 'persp')

  ax.scatter3D(x0, y0, z0, color = "green")
  #ax.quiver(x0,y0,z0,N_x0,N_y0,N_z0,
   #         label = 'SSI Method \n'+f'f = ({frequency})[Hz]\n'+r'$\xi = $'+f'({damping})%')
  
  ax.quiver(x0,y0,z0,N_x0,N_y0,N_z0,
            
            label = 'FDD Method \n'+f'f = {F} [Hz]\n'+r'$\xi = $'+f'{D} %\n'+'SSI Method \n'+ f'f = ({frequency})[Hz]\n'+r'$\xi = $'+f'({damping})%',
            color = 'red')


  if title:  plt.title(f"{title}\n"+f'sensors used: {len(x0)}')
  ax.set_xlabel('X-axis [mm]', fontweight ='bold')
  ax.set_ylabel('Y-axis [mm]', fontweight ='bold')
  ax.set_zlabel('Z-axis [mm]', fontweight ='bold')
  

  # show plot
  cursor = mplcursors.cursor(fig, hover=True)
  plt.legend()
  if name: plt.savefig(rf'{name}_Figure.pdf')
  plt.show()
#%%



#%%
""""
Programar para todo una carpeta
"""
# Ruta del directorio de entrada y salida
#input_dir = str(input())
input_dir = r'C:\Users\Usuario\OneDrive - enpc.fr\PC\ENPC\Stage 2A\Documentos\Datos\USW1\Bruit\Results\Tables'
input_dir = rf'{input_dir}'
output_dir = input_dir + '\Figures 3D'
output_dir = rf'{output_dir}' 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f'Carpeta creada en: {output_dir}')
else:
    print(f'La carpeta ya existe en: {output_dir}')
#%%
data_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".csv")]
data_files = sorted(data_files)

modos_archivos = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".json")]
modos_archivos = sorted(modos_archivos)

#%%
#print(modos_archivos[0])

with open(f'{modos_archivos[0]}', 'r') as archivo:
    modos = json.load(archivo)
 

x0 = np.array(modos['x0'])
y0 = np.array(modos['y0'])
z0 = np.array(modos['z0'])

print(x0)
archivo =  data_files[0]
archivo = pd.read_csv(archivo, header = 0, sep=",", index_col = 0)
f = sp.stats.trim_mean(archivo['f_T'], 0.1)
print(f)
#%%

names = list()
for file in data_files[:]:
    #name = os.path.basename(file)
    name = os.path.splitext(os.path.basename(file))[0]
    names.append(name)
#%%

for file,name,modos_archivo in zip(data_files,names, modos_archivos):
    Figure3D(modos_archivo,file, c = 120,name = f'{output_dir}\{name}', title = f'{name}')
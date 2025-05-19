import numpy as np
from scipy.integrate import simpson as simps
import loadspectra
from readfile import readata
from matplotlib import pyplot as plt

# Função de normalização relativa dos espectros a partir da área
def normspec(spectra, wl, ref=1):
    normspectra = np.zeros_like(spectra)
    arearef = simps(spectra[:, ref], x=wl)
    for i in range(spectra.shape[1]):
        area = simps(spectra[:, i], x=wl)
        normspectra[:, i] = spectra[:, i] * arearef / area
    return np.array(normspectra)

# Função que abre o arquivo de comprimento de ondas:
def wavereader(file_name):
    # ler os comprimentos de onda (TRANSFORMAR EM UMA FUNÇÃO PRÓPRIA)
    file = np.loadtxt(file_name, delimiter='\t', dtype=str) # "C:/Users/admin/Documents/Projeto/Soreteffect-Stokesshift/08-23-2024/morning-tambiente/wavelength.txt"
    file = np.array([file[i].split() for i in range(len(file))])
    wl = np.array([float(x.replace(',', '.')) for x in file[:,0]])
    wn = 1e7/wl
    return wl, wn

# Função que subtrai o branco
def noisub(path, speca, n_speca, base, n_base):
    branco = loadspectra.dataloadop(path, base, n_base) # ler o branco
    spectra = loadspectra.dataloadop(path, speca, n_speca) # ler os espectros
    # faz média do branco
    meanbranco = np.array([float(np.mean(coluna)) for coluna in branco])
    # subtrai branco dos dados
    return spectra - np.array(meanbranco).reshape(-1,1)

# Função de encontra os picos dos espectros
def peakfinder(wl, spectra): # wlfn: nome do arquivo de comprimentos de onda   
    # encontrar os pontos de máximo para cada espectro
    peaks = np.max(spectra, axis=0) # Intensidade máxima de cada espectro
    peak_wavelengths = wl[np.argmax(spectra, axis=0)] # Comprimento de onda correspondente
    return peaks, peak_wavelengths

# Função que cria o header de tempo de medida dos espectros
def timeheader(inttime, tstep, spectra):   
    return np.arange(spectra.shape[1]) * (inttime + tstep) / 60

# SELECIONAR ESPECTROS (t = 5, 15, 25, 35 min):
def spectraselector(spectra, time_row, ti=0, ts=5, qt=5):
    targt = np.array([(ti+k*ts) for k in range(qt)]) # 0) Array com os tempos definidos como referência
    tind = np.abs(time_row - targt[:, None]).argmin(axis=1) # 1) tomar os índices desses tempos
    selected_times = time_row[tind] # 2) encontrar os valores de tempo mais próximos dos tempos desejados
    selected_spectra = spectra[:, tind] # 3) encontrar os espectros relacionados a esses índices
    return selected_times, selected_spectra
    
def tempheader(step, qt, ti=20):    
    # 0) criar o nome dos arquivos e um arquivo com as temperaturas
    return np.array([(ti+step*i) for i in range(qt)]) # temperaturas

# REALIZAR A ANÁLISE DOS DADOS DE ESPECTRO POR TEMPERATURA COM A NORMALIZAÇÃO PELA ÁREA
def tempnorm(tsp, fbn, cc, temps): # tsp: caminho dos arquivos; fbn: nome base dos arquivos; cc: CC.ifx/txt ou CD.ifx/txt ou .ifx/txt
    tempspectra = []
    for i in range(len(temps)):
        filename = tsp + "/" + fbn + str(temps[i]) + cc
        unus1, unus2, wls, tempspectru = readata(filename, 23)
        tempspectra.append(tempspectru)
    wls = np.array(wls)
    tempspectra = np.array(tempspectra).T
    # 2) fazer a normalização dos espectros com o método das áreas
    nts = normspec(tempspectra, wls)
    # 3) utilizar todos os gráficos para fazer a diferença entre eles (T1 - T2, etc.) - tem que ser feito de forma a não criar duplicados (isso já está montado e é só transferir para cá)
    return temps, wls, tempspectra, nts

# Concatenar a linha de tempos de medida com os dados
def adjdata(x, y, z): # x header; y: wavelength; z: data matrix
    x = x.reshape(1, -1)
    y = y.reshape(-1, 1)
    x = np.concatenate([[np.nan], x[0]])
    yz = np.hstack([y, z])
    xyz = np.vstack([x, yz])
    return xyz

# Faz a diferença entre dois espectros na matriz, para todos os espectros
def difspec(k, headername, spectramatrix):
    header = headername[k] # header da referência
    header2 = [] # Header desses gráficos que serão montados
    difference = [] # matriz de diferêncas
    for i in range(spectramatrix.shape[1]): # loop que faz subtração
        if i != k:
            difference.append(spectramatrix[:, k] - spectramatrix[:, i])
            header2.append(str(header)+" - "+str(headername[i]))
    return np.array(header2),np.array(difference)

# TESTE:
x = np.array(np.linspace(1, 5, 5))
y = np.array(np.linspace(1, 10, 10))
z = np.array(np.linspace(20, 100, 50)).reshape(10, 5)
dhead, diff = difspec(0, x, z)
xyz = adjdata(x, y, z)

# TESTE:
path = "C:/Users/admin/Documents/Projeto/Soreteffect-Stokesshift/First-Run"
file_name = "data-range-rb-fs.txt"
wn_name = path+"/wavenumbers.txt"
wn, wl = wavereader(wn_name)
data = loadspectra.dataloadorigintxt(path, file_name)
normspectra = normspec(data, wl)
temps = tempheader(20, 6, 10)


"PASSAR TUDO QUE ESTÁ A PARTIR DAQUI PARA O OUTRO MÓDULO"
# region Espectros dependência com a temperatura
# Cria figura e subplots
fig = plt.figure(figsize=(14, 6))

# --- Gráfico: Espectros normalizados ---
ax1 = fig.add_subplot(121)
for i in range(len(temps)):
    ax1.plot(wl, normspectra[:, i], label=f'{temps[i]:.0f} °C', lw=0.5)

ax1.set_xlabel('comprimento de onda (nm)')
ax1.set_xlim([500,800])
ax1.set_ylabel('Intensidade')
ax1.set_ylim(0)
ax1.set_title('Espectros')
ax1.legend(title='Temperatura')

ax2 = fig.add_subplot(122)
ax2.plot(wl, normspectra[:, 0] - normspectra[:, 5], label=f'20 - 70 °C', lw=0.5)

ax2.set_xlabel('comprimento de onda (nm)')
ax2.set_xlim([500,800])
ax2.set_ylabel('Intensidade')
ax2.set_title('Diferença entre Espectros')
ax2.legend(title='Variação de temperatura')

plt.tight_layout()
plt.show()
#endregion
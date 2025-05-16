import numpy as np
from scipy.integrate import simpson as simps
from loadspectra import dataload
from readfile import readata

# Função de normalização relativa dos espectros a partir da área
def normspec(spectra, wl, ref=1):
    normspectra = np.zeros_like(spectra)
    arearef = simps(spectra[:, ref], x=wl)
    for i in range(spectra.shape[1]):
        area = simps(spectra[:, i], x=wl)
        normspectra[:, i] = spectra[:, i] * arearef / area
    return normspectra

# Função de processamento de dados
def dataproc(p, b, s, tstep, inttime):
    # ler os comprimentos de onda
    data2 = np.loadtxt("C:/Users/admin/Documents/Projeto/Soreteffect-Stokesshift/08-23-2024/morning-tambiente/wavelength.txt", delimiter='\t', dtype=str)
    data2 = np.array([data2[i].split() for i in range(len(data2))])
    wl = np.array([float(x) for x in data2[:,0]])
    wn = 1e7/wl
    # ler os dados
    branco = dataload(p, b, 10)    # ler o branco
    spectra = dataload(p,s, 700) # ler os espectros
    # faz média do branco
    meanbranco = np.array([float(np.mean(coluna)) for coluna in branco])
    # subtrai branco dos dados
    adjspectra = spectra - np.array(meanbranco).reshape(-1,1)
    # criar o header de tempo de medida dos espectros
    ts = inttime + tstep # time step
    time_row = np.arange(adjspectra.shape[1]) * ts / 60
    # fazer normalização com a área da curva
    normspectra = normspec(adjspectra, wl)
    # encontrar os pontos de máximo para cada espectro
    peaks = np.max(normspectra, axis=0) # Intensidade máxima de cada espectro
    peak_wavelengths = wl[np.argmax(normspectra, axis=0)] # Comprimento de onda correspondente
    return wl, wn, adjspectra, ts, time_row, normspectra, peaks, peak_wavelengths

# SELECIONAR ESPECTROS (t = 5, 15, 25, 35 min):
def spectraselector(spectra, time_row, ti=0, ts=5, qt=5):
    targt = np.array([(ti+k*ts) for k in range(qt)]) # 0) Array com os tempos definidos como referência
    tind = np.abs(time_row - targt[:, None]).argmin(axis=1) # 1) tomar os índices desses tempos
    selected_times = time_row[tind] # 2) encontrar os valores de tempo mais próximos dos tempos desejados
    selected_spectra = spectra[:, tind] # 3) encontrar os espectros relacionados a esses índices

    return selected_times, selected_spectra

# REALIZAR A ANÁLISE DOS DADOS DE ESPECTRO POR TEMPERATURA COM A NORMALIZAÇÃO PELA ÁREA
def tempnorm(tsp, fbn, cc, ti, qt, step): # tsp: caminho dos arquivos; fbn: nome base dos arquivos; cc: CC.ifx/txt ou CD.ifx/txt ou .ifx/txt
    # 0) criar o nome dos arquivos e um arquivo com as temperaturas
    temps = np.array([(ti+step*i) for i in range(qt)]) # temperaturas
    # 1) ler os espectros e ajustá-los
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
    print(np.shape(x))
    yz = np.hstack([y, z])
    xyz = np.vstack([x, yz])
    return xyz

# TESTE:
x = np.array(np.linspace(1, 5, 5))
y = np.array(np.linspace(1, 10, 10))
z = np.array(np.linspace(20, 100, 50)).reshape(10, 5)
xyz = adjdata(x, y, z)
print(xyz)
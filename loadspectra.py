import numpy as np
from scipy.integrate import simpson as simps
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from readfile import readata

# Função para ler e ajustar os dados
def dataload(path,basefilename,nspectra=10):
    sp = []
    for i in range(nspectra):
        data = np.loadtxt(path+"/"+basefilename+"_"+str(i)+"s.txt", delimiter='\t', dtype=str) # ler os espectros
        data = np.array([float(x) for x in data])
        sp.append(data)
    sp = np.array(sp).T # faz um array transposto com colunas sendo cada espectro
    return sp

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

# TESTE (SEIKOUDA)
path = "C:/Users/admin/Documents/Projeto/SESS150425"
bnbranco = "B150425"
bnspectra = "AR100425"
tstep = 6
inttime = 1.5
wl, wn, adjspectra, ts, time_row, normspectra, peaks, peak_wavelengths = dataproc(path, bnbranco, bnspectra, tstep, inttime)

# ESCOLHER QUATRO ESPECTROS (t = 5, 15, 25, 35 min):
targt = np.array([5, 15, 25, 35]) # 0) Array com os tempos definidos como referência
tind = np.abs(time_row - targt[:, None]).argmin(axis=1) # 1) tomar os índices desses tempos
selected_times = time_row[tind] # 2) encontrar os valores de tempo mais próximos dos tempos desejados
# 3) encontrar os espectros relacionados a esses índices
selected_raw_spectra = adjspectra[:, tind]         # espectros não normalizados
selected_norm_spectra = normspectra[:, tind]       # espectros normalizados

# REALIZAR A ANÁLISE DOS DADOS DE ESPECTRO POR TEMPERATURA COM A NORMALIZAÇÃO PELA ÁREA
# 0) criar o nome dos arquivos e um arquivo com as temperaturas
temps = np.array([(20+i*5) for i in range(7)]) # temperaturas
tsp = "C:/Users/admin/Documents/Projeto/Fluorescence-spectraxTemperature" # caminho do arquivo
fbn = "RB090425-TV" # nome base dos arquivos
cc = "CC.ifx" # designação final nos arquivos (C de gaus celcius e C de crescente - direção da variação de temperatura)
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


# concatenar a linha de tempos de medida com os dados
complete_data = np.vstack([time_row, adjspectra])

# region Gráfico 3D com os espectros normalizados:
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')

# for i in range(normspectra.shape[1]):
#     y = np.full_like(wl, time_row[i])  # eixo Y = tempo constante
#     cor = 'dimgray' if i % 2 == 0 else 'k' # alterna entre cinza claro e escuro
#     ax.plot(wn, y, normspectra[:, i], label=f'{time_row[i]:.1f} min', color=cor)

# y_min = 0
# y_max = time_row[-1] + ts / 60  # adiciona mais um passo (ts convertido para minutos)

# ax.set_ylim(y_min, y_max)

# ax.set_xlabel('Número de onda (cm⁻¹)')
# ax.set_ylabel('Tempo (min)')
# ax.set_zlabel('Intensidade normalizada')
# plt.title('Espectros normalizados')
# plt.tight_layout()
# plt.show()
# endregion

# region Gráficos 3D + wlXt + IXt:
# # Cria figura e subplots
# fig = plt.figure(figsize=(14, 6))

# # --- Gráfico 3D dos picos ---
# ax1 = fig.add_subplot(131, projection='3d')
# ax1.plot(time_row, peak_wavelengths, peaks, marker='o', linestyle='-', color='black')
# ax1.set_xlabel('Tempo (min)')
# ax1.set_ylabel('Comprimento de onda do pico')
# ax1.set_zlabel('Intensidade máxima')
# ax1.set_title('Pico 3D: Tempo x λ_max x Intensidade')

# # --- Gráfico 2D: Intensidade máxima vs Tempo ---
# ax2 = fig.add_subplot(132)
# ax2.plot(time_row, peaks, color='black')
# ax2.set_xlabel('Tempo (min)')
# ax2.set_ylabel('Intensidade máxima')
# ax2.set_title('Intensidade do Pico vs Tempo')
# ax2.axvline(x=10, color='gray', linestyle='--', linewidth=1)
# ax2.axvline(x=40, color='gray', linestyle='--', linewidth=1)


# # --- Gráfico 2D: Comprimento de onda do pico vs Tempo ---
# ax3 = fig.add_subplot(133)
# ax3.plot(time_row, peak_wavelengths, color='black')
# ax3.set_xlabel('Tempo (min)')
# ax3.set_ylabel('Comprimento de onda do pico')
# ax3.set_title('λ_max vs Tempo')
# ax3.axvline(x=10, color='gray', linestyle='--', linewidth=1)
# ax3.axvline(x=40, color='gray', linestyle='--', linewidth=1)

# plt.tight_layout()
# plt.show()
# endregion

# region Gráfico com os 4 espectros selecionados
# # Cria figura e subplots
# fig = plt.figure(figsize=(14, 6))

# # --- Gráfico: Espectros não normalizados ---
# ax1 = fig.add_subplot(121)
# for i in range(len(selected_times)):
#     ax1.plot(wl, selected_raw_spectra[:, i], label=f'{selected_times[i]:.0f} min', lw=0.5)

# ax1.set_xlabel('comprimento de onda (nm)')
# ax1.set_xlim([500,700])
# ax1.set_ylabel('Intensidade')
# ax1.set_ylim(0)
# ax1.set_title('Não normalizado')
# ax1.legend(title='Tempo da medida')

# # --- Gráfico: Espectros normalizados ---
# ax2 = fig.add_subplot(122)
# for i in range(len(selected_times)):
#     ax2.plot(wl, selected_norm_spectra[:, i], label=f'{selected_times[i]:.0f} min', lw=0.5)

# ax2.set_xlabel('comprimento de onda (nm)')
# ax2.set_xlim([500,700])
# ax2.set_ylabel('Intensidade')
# ax2.set_ylim(0)
# ax2.set_title('Normalizado')
# ax2.legend(title='Tempo da medida')

# plt.tight_layout()
# plt.show()
# endregion

# region Espectros dependência com a temperatura
# Cria figura e subplots
fig = plt.figure(figsize=(14, 6))

# --- Gráfico: Espectros normalizados ---
ax1 = fig.add_subplot(121)
for i in range(len(temps)):
    ax1.plot(wls, nts[:, i], label=f'{temps[i]:.0f} °C', lw=0.5)

ax1.set_xlabel('comprimento de onda (nm)')
ax1.set_xlim([500,800])
ax1.set_ylabel('Intensidade')
ax1.set_ylim(0)
ax1.set_title('Normalizado')
ax1.legend(title='Temperatura')

ax2 = fig.add_subplot(122)
# for i in range(len(temps)):
#     ax2.plot(wls, tempspectra[:, i], label=f'{temps[i]:.0f} °C', lw=0.5)

# ax2.set_xlabel('comprimento de onda (nm)')
# ax2.set_xlim([500,800])
# ax2.set_ylabel('Intensidade')
# ax2.set_ylim(0)
# ax2.set_title('Não normalizado')
# ax2.legend(title='Temperatura')

# --- Gráfico: diferença entre espectros ---
ax2 = fig.add_subplot(122)
ax2.plot(wls, nts[:, 0] - nts[:, len(temps)-1], label=f'20 - 50 °C', lw=0.5)
ax2.set_xlabel('comprimento de onda (nm)')
ax2.set_xlim([500,800])
ax2.set_ylabel('Intensidade')
ax2.set_title('Normalizado')
ax2.legend(title='Temperatura')
#endregion

plt.tight_layout()
plt.show()


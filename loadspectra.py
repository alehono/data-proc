import numpy as np

# Função para ler e ajustar os dados
def dataloadop(path,basefilename,nspectra=10):
    sp = []
    for i in range(nspectra):
        data = np.loadtxt(path+"/"+basefilename+"_"+str(i)+"s.txt", delimiter='\t', dtype=str) # ler os espectros
        data = np.array([float(x) for x in data])
        sp.append(data)
    sp = np.array(sp).T # faz um array transposto com colunas sendo cada espectro
    return sp

def dataloadorigintxt(path, file_name):
    data = np.loadtxt(path+"/"+file_name, delimiter='\t', dtype=str) # ler os espectros
    data = np.array([[float(x.replace(',','.')) for x in line] for line in data])
    return data


"PASSAR TUDO QUE ESTÁ A PARTIR DAQUI PARA O OUTRO MÓDULO"
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
# fig = plt.figure(figsize=(14, 6))

# # --- Gráfico: Espectros normalizados ---
# ax1 = fig.add_subplot(121)
# for i in range(len(temps)):
#     ax1.plot(wls, nts[:, i], label=f'{temps[i]:.0f} °C', lw=0.5)

# ax1.set_xlabel('comprimento de onda (nm)')
# ax1.set_xlim([500,800])
# ax1.set_ylabel('Intensidade')
# ax1.set_ylim(0)
# ax1.set_title('Normalizado')
# ax1.legend(title='Temperatura')

# ax2 = fig.add_subplot(122)
# for i in range(len(temps)):
#     ax2.plot(wls, tempspectra[:, i], label=f'{temps[i]:.0f} °C', lw=0.5)

# ax2.set_xlabel('comprimento de onda (nm)')
# ax2.set_xlim([500,800])
# ax2.set_ylabel('Intensidade')
# ax2.set_ylim(0)
# ax2.set_title('Não normalizado')
# ax2.legend(title='Temperatura')

# --- Gráfico: diferença entre espectros ---
# ax2 = fig.add_subplot(122)
# ax2.plot(wls, nts[:, 0] - nts[:, len(temps)-1], label=f'20 - 50 °C', lw=0.5)
# ax2.set_xlabel('comprimento de onda (nm)')
# ax2.set_xlim([500,800])
# ax2.set_ylabel('Intensidade')
# ax2.set_title('Normalizado')
# ax2.legend(title='Temperatura')


# plt.tight_layout()
# plt.show()
#endregion

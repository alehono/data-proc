from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Monta um layer do gráfico
def layer(i, ngraph, fig, wl, spectra, legends, descriptor, lmin, lmax, xlabel='Comprimento de onda (nm)',
    ylabel='Intensidade',
    title='Espectros',
    legend_title='Temperatura'):
    position = int(f'1{ngraph}{i+1}')

    ax = fig.add_subplot(position)
    for j in range(len(legends)):
        ax.plot(wl, spectra[:, j], label=f'{legends[j]:.0f} {descriptor}', lw=0.5)

    ax.set_xlabel(xlabel)
    if lmin is not None and lmax is not None:
        ax.set_xlim([lmin,lmax])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=legend_title)

# Monta o gráfico
def graphgen(ngraph, wl, spectralist, infolist):
    fig = plt.figure(figsize=(14, 6))
    for i in range(ngraph):
        layer(i, ngraph, fig, wl, spectralist[i], infolist[i,0], infolist[i,1], infolist[i,2], infolist[i,3])

    plt.tight_layout()
    plt.show()
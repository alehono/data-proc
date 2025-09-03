import loadspectra
import procdata
import savefile
import graphhandler
import numpy as np

def main():
    # Espectros em função do tempo
    # Primeira corrida
    path = "C:/Users/admin/Documents/Projeto/Soreteffect-Stokesshift/First-Run"
    file_name = "data-range-rb-fs.txt"
    wn_name = path+"/wavenumbers.txt"
    frunwn, frunwl = procdata.wavereader(wn_name)
    frun = loadspectra.dataloadorigintxt(path, file_name)
    frunnorm = procdata.normspec(frun, frunwl)
    temps = procdata.tempheader(10, frun.shape[1])
    frunxyz = procdata.adjdata(temps, frunwl, frunnorm)
    np.savetxt("results/first_run_area_normalized_TXspectra.txt", frunxyz, delimiter='\t') # fazer a diferença e salvar os dados
    for j in range(frunnorm.shape[1]):
        tempheader, tempdata = procdata.difspec(j, temps, frunnorm)
        na = np.array(['']).reshape(1,-1)
        tempheader = np.hstack([na, tempheader.reshape(1,-1)])
        header = "\t".join((f"{t}C") for t in tempheader[0,:])
        diffdata = procdata.adjdata2(frunwl, tempdata)
        np.savetxt(f'results/first_run_diff_area_normTXspectra{temps[j]}.txt', diffdata, delimiter='\t', header=header, comments='', encoding='utf-8')

    
    # Segunda corrida
    path = "C:/Users/admin/Documents/Projeto/Fluorescence-spectraxTemperature"
    basefilename = "RB090425-TV"
    cc = "CC.ifx"
    tempsheader = procdata.tempheader(5, 7)
    srunwls, srun, srunnorm = procdata.tempnorm(path, basefilename, cc, tempsheader) # wls, tempspectra, nts
    frunxyz = procdata.adjdata(tempsheader, srunwls, srunnorm)
    np.savetxt("results/second_run_area_normalized_TXspectra.txt", frunxyz, delimiter='\t') # salvar os dados
    for j in range(srunnorm.shape[1]): # fazer a diferença
        tempheader, tempdata = procdata.difspec(j, tempsheader, srunnorm)
        na = np.array(['']).reshape(1,-1)
        tempheader = np.hstack([na, tempheader.reshape(1,-1)])
        header = "\t".join((f"{t}C") for t in tempheader[0,:])
        diffdata = procdata.adjdata2(srunwls, tempdata)
        np.savetxt(f'results/second_run_diff_area_normTXspectra{tempsheader[j]}.txt', diffdata, delimiter='\t', header=header, comments='', encoding='utf-8')
    
    # Abrir os espectros em função do tempo (700)
    path = "C:/Users/admin/Documents/Projeto/SESS150425"
    sfilename = "AR100425"
    basefilename = "B150425"
    file_name = "C:/Users/admin/Documents/Projeto/Soreteffect-Stokesshift/08-23-2024/morning-tambiente/wavelength.txt"
    wl, wn = procdata.wavereader(file_name)
    tsp_ns = procdata.noisub(path, sfilename, 700, basefilename, 10) # Abrir espectros (path, speca, n_speca, base, n_base)
    specnorm = procdata.normspec(tsp_ns, wl)
    times = procdata.timeheader(1.5, 6, specnorm)
    trunxyz = procdata.adjdata(times, wl, specnorm)
    np.savetxt("results/time_area_normalized_TXspectra.txt", trunxyz, delimiter='\t') # salvar os dados
    seltimes, selspectra = procdata.spectraselector(specnorm, times, 5, 10)
    selspectraxyz = procdata.adjdata(seltimes, wl, selspectra)
    np.savetxt("results/selected-time_area_normalized_TXspectra.txt", selspectraxyz, delimiter='\t') # salvar os dados
    for j in range(selspectra.shape[1]): # fazer a diferença
        tempheader, tempdata = procdata.difspec(j, seltimes, selspectra)
        na = np.array(['']).reshape(1,-1)
        tempheader = np.hstack([na, tempheader.reshape(1,-1)])
        header = "\t".join((f"{t}min") for t in tempheader[0,:])
        diffdata = procdata.adjdata2(wl, tempdata)
        np.savetxt(f'results/selected_times_diff_area_normTXspectra-{int(seltimes[j])}.txt', diffdata, delimiter='\t', header=header, comments='', encoding='utf-8')

    
main()
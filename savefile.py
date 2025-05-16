def savedata(arquivo, header, bcount, movavg, window):
    with open(arquivo.replace(".txt", "")+"-movavg_w"+str(window)+".csv", "w") as dados:
        dados.write(header[0]+";"+header[1]+"\n")
        for i in range(len(bcount)):
            dados.write(str(bcount[i])+";"+str(movavg[i])+"\n")
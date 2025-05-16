def readata(filename, line=1, hline=0):
    file1 = open(filename, 'r')
    fname = filename
    data = file1.readlines()
    bcount = []
    pcount = []
    try:
        header = data[hline-1].split(' '+'\t') # tem que modificar para encontrar o ponto correto independente de onde ele começa no arquivo
        header[2] = "moving average"
    except:
        header = []
    for i in range(line-1,len(data)): # tem que modificar para encontrar o ponto correto independente de onde ele começa no arquivo
        line = data[i].split()
        bcount.append(float(line[0]))
        pcount.append(float(line[1]))
    file1.close()
    return fname, header, bcount, pcount

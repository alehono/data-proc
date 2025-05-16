def readata(filename, line=1, hline=0):
    file1 = open(filename, 'r')
    fname = filename
    data = file1.readlines()
    bcount = []
    pcount = []
    try:
        header = data[hline-1].split(' '+'\t') 
    except:
        header = []
    for i in range(line-1,len(data)): 
        line = data[i].split()
        bcount.append(float(line[0]))
        pcount.append(float(line[1]))
    file1.close()
    return fname, header, bcount, pcount

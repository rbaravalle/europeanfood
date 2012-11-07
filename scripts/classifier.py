import csv
import sys
import os
from subprocess import *
from gch import colorHistogram

def conf_mat(test, classes):
        m = [ [0 for i in range(max(classes))] for j in range(max(classes))]
        for i in range(len(classes)):
            m[test[i]-1][classes[i]-1] = m[test[i]-1][classes[i]-1] + 1
        return m

# get cross validation of executing ./easy.py ../exps/gchS.txt
def getCross():

    easy = './easy.py'
    scanner = '../exps/gchS.txt'

    cmd = '{0} {1}'.format(easy, scanner)
    print cmd
    f = Popen(cmd, shell = True, stdout = PIPE).stdout

    line = 1
    while True:
        last_line = line
        line = f.readline()
        if str(line).find("rate") != -1:        
            cross = float(line.split()[-1][5:8])
            break
        if not line: break
    return cross

def test():
    arch = './gchC.txt.predict'

    easy = './easy.py'
    scanner = '../exps/gchS.txt'
    camera = '../exps/gchC.txt'

    cmd = '{0} {1} {2}'.format(easy, scanner, camera)
    print cmd
    f = Popen(cmd, shell = True).communicate()
   
    cmd = 'cat "{0}"'.format(arch)
    print cmd
    f = Popen(cmd, shell = True, stdout = PIPE).stdout
    g = f

    c = 0
    line = 1
    while True:
        last_line = line
        line = f.readline()
        c = c+1
        if not line: break
    testL = [0 for i in range(c)]

    cmd = 'cat "{0}"'.format(arch)
    print cmd
    f = Popen(cmd, shell = True, stdout = PIPE).stdout
    c = 0
    line = 1
    while True:
        last_line = line
        line = f.readline()
        testL[c] = int(last_line)
        c = c+1
        if not line: break

    a = [i for i in range(len(testL)-1)]
    a = map(lambda i: i/20+1, a)

    print "Test: ", testL

    b = conf_mat(testL,a)
    for row in b:
        print row

    return testL



def main():
    cant = 20+1
    dDFs  = 64
    baguette = [['Df' for j in range(dDFs)] for i in range(cant)]
    salvado   = [['Df' for j in range(dDFs)] for i in range(cant)]
    lactal   = [['Df' for j in range(dDFs)] for i in range(cant)]
    sandwich = [['Df' for j in range(dDFs)] for i in range(cant)]

    baguetteC = [['Df' for j in range(dDFs)] for i in range(cant)]
    salvadoC   = [['Df' for j in range(dDFs)] for i in range(cant)]
    lactalC   = [['Df' for j in range(dDFs)] for i in range(cant)]
    sandwichC = [['Df' for j in range(dDFs)] for i in range(cant)]

    path = '../images/nonbread/res/'
    dirList=os.listdir(path)
    print len(dirList)
    
    nonbread = [['Df' for j in range(dDFs)] for i in range(len(dirList))]


    import Image
    j = 0
    for i in range(len(dirList)):
        filename = path+dirList[i]
        I = Image.open(filename)
        if(I.mode == 'RGB'):
            print filename
            nonbread[j] = colorHistogram(filename)
            j = j+1
        if (j > 41):
            break

    for i in range(1,cant):
        filename = '../images/scanner/baguette/baguette{}.tif'.format(i)
        print filename
        baguette[i] = colorHistogram(filename)
        filename = '../images/scanner/lactal/lactal{}.tif'.format(i)
        print filename
        lactal[i] = colorHistogram(filename)
        filename = '../images/scanner/salvado/salvado{}.tif'.format(i)
        print filename
        salvado[i] = colorHistogram(filename)
        filename = '../images/scanner/sandwich/sandwich{}.tif'.format(i)
        print filename
        sandwich[i] = colorHistogram(filename)


        v = 50
        b = 1.05
        filename = '../images/camera/baguette/slicer/b{}.tif'.format(i)
        print filename
        baguetteC[i] = colorHistogram(filename)
        filename = '../images/camera/lactal/l{}.tif'.format(i)
        print filename
        lactalC[i] = colorHistogram(filename)
        filename = '../images/camera/salvado/s{}.tif'.format(i)
        print filename
        salvadoC[i] = colorHistogram(filename)
        filename = '../images/camera/sandwich/s{}.tif'.format(i)
        print filename
        sandwichC[i] = colorHistogram(filename)
   

    with open('../exps/gchS.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(baguette[1:]+lactal[1:]+salvado[1:]+sandwich[1:]+nonbread[0:20])

    with open('../exps/gchC.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(baguetteC[1:]+lactalC[1:]+salvadoC[1:]+sandwichC[1:]+nonbread[20:40])

    prog = './a.out' # convert.c
    csvS = '../exps/gchS.csv'
    csvC = '../exps/gchC.csv'
    txtS = '../exps/gchS.txt'
    txtC = '../exps/gchC.txt'
    cmd = '{0} "{1}" > "{2}"'.format(prog, csvS, txtS)
    Popen(cmd, shell = True, stdout = PIPE).communicate()	
    cmd = '{0} "{1}" > "{2}"'.format(prog, csvC, txtC)
    Popen(cmd, shell = True, stdout = PIPE).communicate()
    test()


main()

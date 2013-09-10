import csv
import sys
import os
from subprocess import *
from gch import colorHistogram
#import pyccv
import colortransforms
import numpy as np
from featureTexture import *
#import singularityEntropyCL as sg
#import localmfrac
import time
import cielab
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import mfs
#import efd
import gradient
import laplacian
#import singularityMFS as smfs
from matplotlib import pyplot as plt
from vf import VF
from mca import mca
import matplotlib.pyplot as plt
from pylab import *
import scipy

SVM = 0
RANDOM_FOREST = 1
cantDF = 10

def callF(filename,which,extra):
    return features(filename,which,3,False,extra)

def features(filename,i,j,combine,extra):
    #farr = [colorHistogram,ccv,haralick,lbp,tas,zernike, singularity.spec]
    #farr = [localmfrac.localMF]
    #farr = [cielab.lab]
    #farr = [lbp]
    #farr = [sg.spec]
    #farr = [smfs.spec]
    #farr = [mfs.mfs]
    #farr = [mfs.mfs, gradient.main]
    #farr = [mfs.mfs, gradient.main, laplacian.laplacian]
    #farr = [mfs.mfs, laplacian.laplacian]
    farr = [mfs.mfs]
    #farr = [gradient.main]
    #farr = [efd.efd]
    #farr = [haralick]
    #farr = [lbp]
    #if(combine==True):
    #    return hstack((farr[1](filename),farr[2](filename),farr[3](filename),farr[4](filename),farr[5](filename)))
    t =  time.clock()
    # num of FDs , Open Image?,  convert to grayscale?, (cielab) use L,a,b?
    cantDF = 10
    extra = [1,cantDF*2,3,True,True]
    res = farr[0](filename, extra)
    #res2 = farr[1](filename,extra)
    #res2 = farr[1](filename,extra)
    #res3 = farr[2](filename,extra)
    t =  time.clock() - t
    #print "Time: ", t
    return res
    #return np.hstack([res,res2])
    #return np.hstack([res,res2,res3])


def ccv(filename):
    import Image
    I = Image.open(filename)
    I.putdata(colortransforms.rgb_to_cielab_i(I))
    size = I.size[0] # Normalize image size
    threshold = 500
    return pyccv.calc_ccv(I,size,threshold)

def conf_mat(test, classes):
    m = [ [0 for i in range(max(classes))] for j in range(max(classes))]
    for i in range(len(classes)):
        m[test[i]-1][classes[i]-1] = m[test[i]-1][classes[i]-1] + 1
    return m

# get cross validation of executing ./easy.py ../exps/gchS.txt
def getCross(fileStxt):

    easy = './easy.py'

    cmd = '{0} {1}'.format(easy, fileStxt)
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


def csvm(dtrain,labels,fileStxt, base):
    arch = base+'.predict'

    easy = './easy.py'

    cmd = '{0} {1}'.format(easy, fileStxt)
    print cmd
    f = Popen(cmd, shell = True).communicate()
   
    
    #return testL

from sklearn.neighbors import KNeighborsClassifier
def cnearestneighbors(data,labels,fileStxt,base):
    cnn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=True)

    scores = cross_validation.cross_val_score(cnn, data, labels, cv=4)
    print scores
    print "Nearest Neighbors: " + str( np.array(scores).mean() )

def crandomforest(data,labels,fileStxt,base):
    #read in data, parse into training and target sets
    #dataset = np.genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]
    #In this case we'll use a random forest, but this could be any classifier
    cfr = RandomForestClassifier(n_estimators=100)

    #Simple K-Fold cross validation. 5 folds.
    #cv = cross_validation.KFold(len(data), k=4, indices=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    #data = np.array(data)
    #labels=np.array(labels)
    scores = cross_validation.cross_val_score(cfr, data, labels, cv=4)
    print scores
#    for traincv, testcv in cv:
#        print "T1, T2", traincv, testcv
#        probas = cfr.fit(data[traincv], labels[traincv]).predict_proba(data[testcv])
#        results.append( logloss.llfun(labels[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    print "Random Forest: " + str( np.array(scores).mean() )


def classifierPredict(dtrain,labels,fileStxt, base, i):
    pred = [csvm, crandomforest, cnearestneighbors]
    print labels
    return pred[i](dtrain,labels,fileStxt, base)


def test(dtrain,labels,fileStxt, base, classifier):
    testL = classifierPredict(dtrain,labels,fileStxt, base,0)
    testL = classifierPredict(dtrain,labels,fileStxt, base,1)
    testL = classifierPredict(dtrain,labels,fileStxt, base,2)



def main(subname,which,local,classifier):
    fsize = 15
    base = subname+'C.txt'
    fileStxt = '../exps/'+subname+'S.txt'
    fileScsv = '../exps/'+subname+'S.csv'
    #fileCtxt = '../exps/'+base
    #fileCcsv = '../exps/'+subname+'C.csv'
    cant = 20
    dDFs  = 20
    baguette = np.zeros((cant, dDFs)).astype(np.float32)
    salvado   = np.zeros((cant, dDFs)).astype(np.float32)
    lactal   = np.zeros((cant, dDFs)).astype(np.float32)
    sandwich = np.zeros((cant, dDFs)).astype(np.float32)
    allied = np.zeros((cant, dDFs)).astype(np.float32)
    doves = np.zeros((cant, dDFs)).astype(np.float32)

    dovesC = np.zeros((cant, dDFs)).astype(np.float32)
    alliedC = np.zeros((cant, dDFs)).astype(np.float32)
    baguetteC = np.zeros((cant, dDFs)).astype(np.float32)
    salvadoC   = np.zeros((cant, dDFs)).astype(np.float32)
    lactalC   = np.zeros((cant, dDFs)).astype(np.float32)
    sandwichC = np.zeros((cant, dDFs)).astype(np.float32)

    vfBaguette = np.zeros(cant).astype(np.float32)
    vfBaguetteC = np.zeros(cant).astype(np.float32)
    mcaBaguette = np.zeros(cant).astype(np.float32)
    mcaBaguetteC = np.zeros(cant).astype(np.float32)
    mcadvBaguette = np.zeros(cant).astype(np.float32)
    mcadvBaguetteC = np.zeros(cant).astype(np.float32)

    vfLactal = np.zeros(cant).astype(np.float32)
    vfLactalC = np.zeros(cant).astype(np.float32)
    mcaLactal = np.zeros(cant).astype(np.float32)
    mcaLactalC = np.zeros(cant).astype(np.float32)
    mcadvLactal = np.zeros(cant).astype(np.float32)
    mcadvLactalC = np.zeros(cant).astype(np.float32)

    vfSalvado = np.zeros(cant).astype(np.float32)
    vfSalvadoC = np.zeros(cant).astype(np.float32)
    mcaSalvado = np.zeros(cant).astype(np.float32)
    mcaSalvadoC = np.zeros(cant).astype(np.float32)
    mcadvSalvado = np.zeros(cant).astype(np.float32)
    mcadvSalvadoC = np.zeros(cant).astype(np.float32)

    vfSandwich = np.zeros(cant).astype(np.float32)
    vfSandwichC = np.zeros(cant).astype(np.float32)
    mcaSandwich = np.zeros(cant).astype(np.float32)
    mcaSandwichC = np.zeros(cant).astype(np.float32)
    mcadvSandwich = np.zeros(cant).astype(np.float32)
    mcadvSandwichC = np.zeros(cant).astype(np.float32)

    vfAllied = np.zeros(cant).astype(np.float32)
    vfAlliedC = np.zeros(cant).astype(np.float32)
    mcaAllied = np.zeros(cant).astype(np.float32)
    mcaAlliedC = np.zeros(cant).astype(np.float32)
    mcadvAllied = np.zeros(cant).astype(np.float32)
    mcadvAlliedC = np.zeros(cant).astype(np.float32)

    vfDoves = np.zeros(cant).astype(np.float32)
    vfDovesC = np.zeros(cant).astype(np.float32)
    mcaDoves = np.zeros(cant).astype(np.float32)
    mcaDovesC = np.zeros(cant).astype(np.float32)
    mcadvDoves = np.zeros(cant).astype(np.float32)
    mcadvDovesC = np.zeros(cant).astype(np.float32)



    path = '../images/nonbread/res/'
    dirList=os.listdir(path)
        
    nonbread = np.zeros((len(dirList), dDFs)).astype(np.float32)

    #read from csv

    with open('../exps/baguetteS.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            baguette[i] = row
            i = i+1

    with open('../exps/baguetteC.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            baguetteC[i] = row
            i = i+1

    with open('../exps/lactalS.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            lactal[i] = row
            i = i+1

    with open('../exps/lactalC.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            lactalC[i] = row
            i = i+1

    with open('../exps/salvadoS.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            salvado[i] = row
            i = i+1

    with open('../exps/salvadoC.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            salvadoC[i] = row
            i = i+1

    with open('../exps/sandwichS.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            sandwich[i] = row
            i = i+1

    with open('../exps/sandwichC.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            sandwichC[i] = row
            i = i+1

    with open('../exps/nonbread.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            nonbread[i] = row
            i = i+1
            if(i > 40): break

    with open('../exps/alliedC.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            alliedC[i] = row
            i = i+1

    with open('../exps/allied.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            allied[i] = row
            i = i+1

    with open('../exps/dovesC.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            dovesC[i] = row
            i = i+1

    with open('../exps/doves.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        for row in spamreader:
            doves[i] = row
            i = i+1


    with open('../exps/vfs.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        vfBaguette = np.array(spamreader.next()).astype(np.float32)
        vfBaguetteC = np.array(spamreader.next()).astype(np.float32)
        vfLactal = np.array(spamreader.next()).astype(np.float32)
        vfLactalC = np.array(spamreader.next()).astype(np.float32)
        vfSalvado = np.array(spamreader.next()).astype(np.float32)
        vfSalvadoC = np.array(spamreader.next()).astype(np.float32)
        vfSandwich = np.array(spamreader.next()).astype(np.float32)
        vfSandwichC = np.array(spamreader.next()).astype(np.float32)

    with open('../exps/mcas.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        mcaBaguette = np.array(spamreader.next()).astype(np.float32)
        mcaBaguetteC = np.array(spamreader.next()).astype(np.float32)
        mcaLactal = np.array(spamreader.next()).astype(np.float32)
        mcaLactalC = np.array(spamreader.next()).astype(np.float32)
        mcaSalvado = np.array(spamreader.next()).astype(np.float32)
        mcaSalvadoC = np.array(spamreader.next()).astype(np.float32)
        mcaSandwich = np.array(spamreader.next()).astype(np.float32)
        mcaSandwichC = np.array(spamreader.next()).astype(np.float32)

    with open('../exps/mcadvs.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        mcadvBaguette = np.array(spamreader.next()).astype(np.float32)
        mcadvBaguetteC = np.array(spamreader.next()).astype(np.float32)
        mcadvLactal = np.array(spamreader.next()).astype(np.float32)
        mcadvLactalC = np.array(spamreader.next()).astype(np.float32)
        mcadvSalvado = np.array(spamreader.next()).astype(np.float32)
        mcadvSalvadoC = np.array(spamreader.next()).astype(np.float32)
        mcadvSandwich = np.array(spamreader.next()).astype(np.float32)
        mcadvSandwichC = np.array(spamreader.next()).astype(np.float32)


    with open('../exps/dovesData.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        vfDoves = np.array(spamreader.next()).astype(np.float32)
        mcaDoves = np.array(spamreader.next()).astype(np.float32)
        mcadvDoves = np.array(spamreader.next()).astype(np.float32)

    with open('../exps/dovesDataC.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        vfDovesC = np.array(spamreader.next()).astype(np.float32)
        mcaDovesC = np.array(spamreader.next()).astype(np.float32)
        mcadvDovesC = np.array(spamreader.next()).astype(np.float32)

    with open('../exps/alliedData.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        vfAllied = np.array(spamreader.next()).astype(np.float32)
        mcaAllied = np.array(spamreader.next()).astype(np.float32)
        mcadvAllied = np.array(spamreader.next()).astype(np.float32)

    with open('../exps/alliedDataC.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        i = 0
        vfAlliedC = np.array(spamreader.next()).astype(np.float32)
        mcaAlliedC = np.array(spamreader.next()).astype(np.float32)
        mcadvAlliedC = np.array(spamreader.next()).astype(np.float32)

    #sandwich = allied
    #sandwichC = alliedC

    #salvado = doves
    #salvadoC = dovesC

    print "ALLIED: ", allied
    print "ALLIEDC: ", alliedC

    #vfSalvado = vfAllied
    #mcaSalvado = mcaAllied
    #mcadvSalvado = mcadvAllied
    #vfSalvadoC = vfAlliedC
    #mcaSalvadoC = mcaAlliedC
    #mcadvSalvadoC = mcadvAlliedC

    #vfSalvado = vfAllied
    #mcaSalvado = mcaAllied
    #mcadvSalvado = mcadvAllied
    #vfSalvadoC = vfAlliedC
    #mcaSalvadoC = mcaAlliedC
    #mcadvSalvadoC = mcadvAlliedC

    data = np.vstack((baguette, baguetteC,lactal,lactalC,salvado,salvadoC,sandwich,sandwichC,nonbread[0:(2*(cant))]))

    #exit()

    #plt.xlabel('Baguette',fontsize=fsize)
    #plt.boxplot(np.vstack((baguette)))
    #plt.show()

    #plt.xlabel('Lactal',fontsize=fsize)
    #plt.boxplot(np.vstack((lactal)))
    #plt.show()

    #plt.xlabel('Salvado',fontsize=fsize)
    #plt.boxplot(np.vstack((salvado)))
    #plt.show()

    #plt.xlabel('Sandwich',fontsize=fsize)
    #plt.boxplot(np.vstack((sandwich)))
    #plt.show()

    #plt.xlabel('BaguetteC',fontsize=fsize)
    #plt.boxplot(np.vstack((baguetteC)))
    #plt.show()

    #plt.xlabel('LactalC',fontsize=fsize)
    #plt.boxplot(np.vstack((lactalC)))
    #plt.show()

    #plt.xlabel('SalvadoC',fontsize=fsize)
    #plt.boxplot(np.vstack((salvadoC)))
    #plt.show()

    #plt.xlabel('SandwichC',fontsize=fsize)
    #plt.boxplot(np.vstack((sandwichC)))
    #plt.show()
    xt = np.arange(1,20,2)
    print xt
    alpha = [   6.,   19.,   32.,   45.,   58.,   71.,   84.,   97.,  110.,  123. , 136. , 149.,
  162.,  175.,  188.,  201. , 214. , 227. , 240.  ,253.]
    alpha = map(lambda i:"%.2f" % float(i/255.), alpha)

    alpha = alpha[0:len(alpha):2]
    print alpha

    rB = plt.boxplot(np.vstack((baguette,baguetteC)), sym='')
    xticks(xt,alpha) # translate
    plt.ylim((0, 2))
    plt.ylabel(r'$f(\alpha)$',fontsize=fsize)
    plt.xlabel(r'$\alpha$',fontsize=fsize)
    mediansB = map(lambda i: i.get_data()[1][0],rB['medians'])
    x = np.arange(len(mediansB))
    plt.plot(map(lambda i: i+1, x), mediansB, 'k+--', label='baguette',linewidth=2.0)
    plt.show()

    #exit()
    plt.ylim((0, 2))
    rL = plt.boxplot(np.vstack((lactal,lactalC)), sym='')
    xticks(xt,alpha) # translate
    plt.ylabel(r'$f(\alpha)$',fontsize=fsize)
    plt.xlabel(r'$\alpha$',fontsize=fsize)
    mediansL = map(lambda i: i.get_data()[1][0],rL['medians'])
    plt.plot(map(lambda i: i+1, x), mediansL, 'k+--', label='sliced',linewidth=2.0)
    plt.show()

    plt.ylim((0, 2))
    rSal = plt.boxplot(np.vstack((salvado,salvadoC)), sym='')
    xticks(xt,alpha) # translate
    mediansSal = map(lambda i: i.get_data()[1][0],rSal['medians'])
    plt.ylabel(r'$f(\alpha)$',fontsize=fsize)
    plt.xlabel(r'$\alpha$',fontsize=fsize)
    plt.plot(map(lambda i: i+1, x), mediansSal, 'k+--',  label='bran',linewidth=2.0)
    plt.show()

    plt.ylim((0, 2))
    rSan = plt.boxplot(np.vstack((sandwich,sandwichC)), sym='')
    xticks(xt,alpha) # translate
    plt.ylabel(r'$f(\alpha)$',fontsize=fsize)
    plt.xlabel(r'$\alpha$',fontsize=fsize)
    mediansSan = map(lambda i: i.get_data()[1][0],rSan['medians'])
    plt.plot(map(lambda i: i+1, x), mediansSan, 'k+--',  label='sandwich',linewidth=2.0)
    plt.show()

    plt.ylim((0, 2))
    rnonB = plt.boxplot(nonbread[0:(2*(cant))], sym='')
    xticks(xt,alpha) # translate
    plt.ylabel(r'$NONBREAD$',fontsize=fsize)
    plt.xlabel(r'$\alpha$',fontsize=fsize)
    mediansNonB = map(lambda i: i.get_data()[1][0],rnonB['medians'])
    plt.plot(map(lambda i: i+1, x), mediansNonB, 'k+--', label='nonbread!',linewidth=2.0)
    plt.show()

    dataB = np.vstack((baguette, baguetteC))
    dataL = np.vstack((lactal,lactalC))
    dataSal = np.vstack((salvado,salvadoC))
    dataSan = np.vstack((sandwich,sandwichC))
    dataTodos = np.vstack((dataB,dataL,dataSal,dataSan))

    # Void Fraction
    arrB = np.hstack((vfBaguette,vfBaguetteC))
    arrL = np.hstack((vfLactal,vfLactalC))
    arrSal = np.hstack((vfSalvado,vfSalvadoC))
    arrSan = np.hstack((vfSandwich,vfSandwichC))
    arrTodos = np.hstack((arrB,arrL,arrSal,arrSan))

    # Mean cell area
    mcaB = np.hstack((mcaBaguette,mcaBaguetteC))
    mcaL = np.hstack((mcaLactal,mcaLactalC))
    mcaSal = np.hstack((mcaSalvado,mcaSalvadoC))
    mcaSan = np.hstack((mcaSandwich,mcaSandwichC))
    mcaTodos = np.hstack((mcaB,mcaL,mcaSal,mcaSan))

    # stdev of Mean cell area
    mcadvB = np.hstack((mcadvBaguette,mcadvBaguetteC))
    mcadvL = np.hstack((mcadvLactal,mcadvLactalC))
    mcadvSal = np.hstack((mcadvSalvado,mcadvSalvadoC))
    mcadvSan = np.hstack((mcadvSandwich,mcadvSandwichC))
    mcadvTodos = np.hstack((mcadvB,mcadvL,mcadvSal,mcadvSan))

    # correlation coefficients
    cfeat = len(dataB[0])
    print cfeat
    cB = np.zeros(cfeat)
    cL = np.zeros(cfeat)
    cSal = np.zeros(cfeat)
    cSan = np.zeros(cfeat)
    cTodos = np.zeros(cfeat)
    cmcaB = np.zeros(cfeat)
    cmcaL = np.zeros(cfeat)
    cmcaSal = np.zeros(cfeat)
    cmcaSan = np.zeros(cfeat)
    cmcaTodos = np.zeros(cfeat)
    cmcadvB = np.zeros(cfeat)
    cmcadvL = np.zeros(cfeat)
    cmcadvSal = np.zeros(cfeat)
    cmcadvSan = np.zeros(cfeat)
    cmcadvTodos = np.zeros(cfeat)
    print "Shapes:"
    print arrTodos.shape
    print dataTodos.shape
    #for i in range(cfeat):
        #print i
       # cB[i] = np.corrcoef(dataB[:,i],arrB)[0,1]
       # cmcaB[i] = np.corrcoef(dataB[:,i],mcaB)[0,1]
       # cmcadvB[i] = np.corrcoef(dataB[:,i],mcadvB)[0,1]
       # cL[i] = np.corrcoef(dataL[:,i],arrL)[0,1]
       # cmcaL[i] = np.corrcoef(dataL[:,i],mcaL)[0,1]
       # cmcadvL[i] = np.corrcoef(dataL[:,i],mcadvL)[0,1]
       # cSal[i] = np.corrcoef(dataSal[:,i],arrSal)[0,1]
       # cmcaSal[i] = np.corrcoef(dataSal[:,i],mcaSal)[0,1]
       # cmcadvSal[i] = np.corrcoef(dataSal[:,i],mcadvSal)[0,1]
       # cSan[i] = np.corrcoef(dataSan[:,i],arrSan)[0,1]
       # cmcaSan[i] = np.corrcoef(dataSan[:,i],mcaSan)[0,1]
       # cmcadvSan[i] = np.corrcoef(dataSan[:,i],mcadvSan)[0,1]
       # cTodos[i] = np.corrcoef(dataTodos[:,i],arrTodos)[0,1]
       # cmcaTodos[i] = np.corrcoef(dataTodos[:,i],mcaTodos)[0,1]
       # cmcadvTodos[i] = np.corrcoef(dataTodos[:,i],mcadvTodos)[0,1]

    cBs = np.zeros(dDFs).astype(np.float32)
    cmcaBs = np.zeros(dDFs).astype(np.float32)
    cmcadvBs = np.zeros(dDFs).astype(np.float32)
    cLs = np.zeros(dDFs).astype(np.float32)
    cmcaLs = np.zeros(dDFs).astype(np.float32)
    cmcadvLs = np.zeros(dDFs).astype(np.float32)
    cSals = np.zeros(dDFs).astype(np.float32)
    cmcaSals = np.zeros(dDFs).astype(np.float32)
    cmcadvSals = np.zeros(dDFs).astype(np.float32)
    cSans = np.zeros(dDFs).astype(np.float32)
    cmcaSans = np.zeros(dDFs).astype(np.float32)
    cmcadvSans = np.zeros(dDFs).astype(np.float32)

    cAs = np.zeros(dDFs).astype(np.float32)
    cmcaAs = np.zeros(dDFs).astype(np.float32)
    cmcadvAs = np.zeros(dDFs).astype(np.float32)

    cBc = np.zeros(dDFs).astype(np.float32)
    cmcaBc = np.zeros(dDFs).astype(np.float32)
    cmcadvBc = np.zeros(dDFs).astype(np.float32)
    cLc = np.zeros(dDFs).astype(np.float32)
    cmcaLc = np.zeros(dDFs).astype(np.float32)
    cmcadvLc = np.zeros(dDFs).astype(np.float32)
    cSalc = np.zeros(dDFs).astype(np.float32)
    cmcaSalc = np.zeros(dDFs).astype(np.float32)
    cmcadvSalc = np.zeros(dDFs).astype(np.float32)
    cSanc = np.zeros(dDFs).astype(np.float32)
    cmcaSanc = np.zeros(dDFs).astype(np.float32)
    cmcadvSanc = np.zeros(dDFs).astype(np.float32)

    cAc = np.zeros(dDFs).astype(np.float32)
    cmcaAc = np.zeros(dDFs).astype(np.float32)
    cmcadvAc = np.zeros(dDFs).astype(np.float32)


    
    #print "All: ", allied
#    print "BAGUETTE", vfBaguette
 #   exit()

    for i in range(dDFs):
        cBs[i] = scipy.stats.stats.spearmanr(baguette[:,i],vfBaguette)[0]
        cmcaBs[i] = scipy.stats.stats.spearmanr(baguette[:,i],mcaBaguette)[0]
        cmcadvBs[i] = scipy.stats.stats.spearmanr(baguette[:,i],mcadvBaguette)[0]
        cLs[i] = scipy.stats.stats.spearmanr(lactal[:,i],vfLactal)[0]
        cmcaLs[i] = scipy.stats.stats.spearmanr(lactal[:,i],mcaLactal)[0]
        cmcadvLs[i] = scipy.stats.stats.spearmanr(lactal[:,i],mcadvLactal)[0]
        cSals[i] = scipy.stats.stats.spearmanr(salvado[:,i],vfSalvado)[0]
        cmcaSals[i] = scipy.stats.stats.spearmanr(salvado[:,i],mcaSalvado)[0]
        cmcadvSals[i] = scipy.stats.stats.spearmanr(salvado[:,i],mcadvSalvado)[0]
        cSans[i] = scipy.stats.stats.spearmanr(sandwich[:,i],vfSandwich)[0]
        cmcaSans[i] = scipy.stats.stats.spearmanr(sandwich[:,i],mcaSandwich)[0]
        cmcadvSans[i] = scipy.stats.stats.spearmanr(sandwich[:,i],mcadvSandwich)[0]
        cAs[i] = scipy.stats.stats.spearmanr(allied[:,i],vfAllied)[0]
        cmcaAs[i] = scipy.stats.stats.spearmanr(allied[:,i],mcaAllied)[0]
        cmcadvAs[i] = scipy.stats.stats.spearmanr(allied[:,i],mcadvAllied)[0]
        #camera
        cBc[i] = scipy.stats.stats.spearmanr(baguetteC[:,i],vfBaguetteC)[0]
        cmcaBc[i] = scipy.stats.stats.spearmanr(baguetteC[:,i],mcaBaguetteC)[0]
        cmcadvBc[i] = scipy.stats.stats.spearmanr(baguetteC[:,i],mcadvBaguetteC)[0]
        cLc[i] = scipy.stats.stats.spearmanr(lactalC[:,i],vfLactalC)[0]
        cmcaLc[i] = scipy.stats.stats.spearmanr(lactalC[:,i],mcaLactalC)[0]
        cmcadvLc[i] = scipy.stats.stats.spearmanr(lactalC[:,i],mcadvLactalC)[0]
        cSalc[i] = scipy.stats.stats.spearmanr(salvadoC[:,i],vfSalvadoC)[0]
        cmcaSalc[i] = scipy.stats.stats.spearmanr(salvadoC[:,i],mcaSalvadoC)[0]
        cmcadvSalc[i] = scipy.stats.stats.spearmanr(salvadoC[:,i],mcadvSalvadoC)[0]
        cSanc[i] = scipy.stats.stats.spearmanr(sandwichC[:,i],vfSandwichC)[0]
        cmcaSanc[i] = scipy.stats.stats.spearmanr(sandwichC[:,i],mcaSandwichC)[0]
        cmcadvSanc[i] = scipy.stats.stats.spearmanr(sandwichC[:,i],mcadvSandwichC)[0]

        cAc[i] = np.corrcoef(alliedC[:,i],vfAlliedC)[0,1]
        cmcaAc[i] = np.corrcoef(alliedC[:,i],mcaAlliedC)[0,1]
        cmcadvAc[i] = np.corrcoef(alliedC[:,i],mcadvAlliedC)[0,1]


    print "Coefficients"
    print "VF Baguette"
    print cB
    print "MCA Baguette"
    print cmcaB
    print "stdev MCA Baguette"
    print cmcadvB
    print "VF Lactal"
    print cL
    print "MCA Lactal"
    print cmcaL
    print "stdev MCA Lactal"
    print cmcadvL
    print "VF Salvado"
    print cSal
    print "MCA Salvado"
    print cmcaSal
    print "stdev MCA Salvado"
    print cmcadvSal
    print "VF Sandwich"
    print cSan
    print "MCA Sandwich"
    print cmcaSan
    print "stdev MCA Sandwich"
    print cmcadvSan
    print "VF Todos"
    print cTodos
    print "MCA Todos"
    print cmcaTodos
    print "stdev MCA Todos"
    print cmcadvTodos


    mean = np.zeros((5,cfeat))
    std = np.zeros((5,cfeat))

    mean[0] = dataB.mean(axis=0)
    std[0] = dataB.std(axis=0)

    mean[1] = dataL.mean(axis=0)
    std[1] = dataL.std(axis=0)

    mean[2] = dataSal.mean(axis=0)
    std[2] = dataSal.std(axis=0)

    mean[3] = dataSan.mean(axis=0)
    std[3] = dataSan.std(axis=0)

    mean[4] = dataTodos.mean(axis=0)
    std[4] = dataTodos.std(axis=0)


    print "Mean : ", mean[0]
    print "Std : ", std[0]
    print "Mean : ", mean[1]
    print "Std : ", std[1]
    print "Mean : ", mean[2]
    print "Std : ", std[2]
    print "Mean : ", mean[3]
    print "Std : ", std[3]
    print "Mean : ", mean[4]
    print "Std : ", std[4]

    x1 = 1
    x2 = 20
    x = np.arange(cfeat)+1
    plt.ylim((-1, 1))
    plt.xlim(x1,x2)
    plt.ylabel(r'$R$',fontsize=fsize)
    plt.xlabel('FD',fontsize=fsize)
    plt.plot(x, cBs, 'ko--', label='baguette',linewidth=2.0)
    plt.plot(x, cLs, 'ro--',  label='sliced',linewidth=2.0)
    plt.plot(x, cSals, 'bo--',  label='bran',linewidth=2.0)
    plt.plot(x, cSans, 'go--',  label='sandwich',linewidth=2.0)
    #plt.plot(x, cTodos, 'mo--',  label='todos',linewidth=2.0)
    plt.legend(loc = 3) # loc 4: bottom, right
    plt.show()

    plt.ylabel(r'$R$',fontsize=fsize)
    plt.xlabel('FD',fontsize=fsize)
    plt.ylim((-1, 1))
    plt.xlim(x1,x2)
    plt.plot(x, cmcaBs, 'ko--', label='baguette',linewidth=2.0)
    plt.plot(x, cmcaLs, 'ro--',  label='sliced',linewidth=2.0)
    plt.plot(x, cmcaSals, 'bo--',  label='bran',linewidth=2.0)
    plt.plot(x, cmcaSans, 'go--',  label='sandwich',linewidth=2.0)
    #plt.plot(x, cmcaTodos, 'mo--',  label='todos',linewidth=2.0)
    plt.legend(loc = 3) # loc 4: bottom, right
    plt.show()

    plt.ylabel(r'$R$',fontsize=fsize)
    plt.xlabel('FD',fontsize=fsize)
    plt.ylim((-1, 1))
    plt.xlim(x1,x2)
    plt.plot(x, cmcadvBs, 'ko--', label='baguette',linewidth=2.0)
    plt.plot(x, cmcadvLs, 'ro--',  label='sliced',linewidth=2.0)
    plt.plot(x, cmcadvSals, 'bo--',  label='bran',linewidth=2.0)
    plt.plot(x, cmcadvSans, 'go--',  label='sandwich',linewidth=2.0)
    #plt.plot(x, cmcadvTodos, 'mo--',  label='todos',linewidth=2.0)
    plt.legend(loc = 3) # loc 4: bottom, right
    plt.show()

    # Graph for means
    plt.xlim(x1,x2)
    x = np.arange(cfeat)+1
    plt.ylim((0, 2))
    xticks(xt,alpha) # translate
    plt.ylabel(r'$mean$',fontsize=fsize)
    plt.xlabel(r'$\alpha$',fontsize=fsize)
    plt.plot(x, mean[0], 'ko--', label='baguette',linewidth=2.0)
    plt.plot(x, mean[1], 'ro--',  label='sliced',linewidth=2.0)
    plt.plot(x, mean[2], 'bo--',  label='bran',linewidth=2.0)
    plt.plot(x, mean[3], 'go--',  label='sandwich',linewidth=2.0)
    #plt.plot(x, mean[4], 'mo--',  label='todos',linewidth=2.0)
    plt.legend(loc = 2)
    plt.show()

    # Graph for standard deviations
    x = np.arange(cfeat)+1
    xticks(xt,alpha) # translate
    plt.ylabel(r'$std$',fontsize=fsize)
    plt.xlabel(r'$\alpha$',fontsize=fsize)
    plt.xlim(x1,x2)
    plt.plot(x, std[0], 'ko--', label='baguette',linewidth=2.0)
    plt.plot(x, std[1], 'ro--',  label='sliced',linewidth=2.0)
    plt.plot(x, std[2], 'bo--',  label='bran',linewidth=2.0)
    plt.plot(x, std[3], 'go--',  label='sandwich',linewidth=2.0)
    #plt.plot(x, std[4], 'mo--',  label='todos',linewidth=2.0)
    plt.legend(loc = 1)
    plt.show()

    with open('means.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(mean)

    with open('stds.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(std)


    #labels = [i for i in range(len(data))]
    labels = np.zeros((len(data),1)) # FIX ME
    for i in range(len(data)):
        labels[i] = i
    labels = map(lambda i: np.floor(i/(2*(cant)))+1, labels)

    print "200?: ", len(data)

    labels = np.array(labels)
    print data
    #print "Labels shape: ", labels.shape
    #print "data shape: ", data.shape
    #data2 = np.hstack((labels,data))
    with open(fileScsv, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    prog = './clas2' # convert.c (here labels are added for the svm)
    cmd = '{0} "{1}" > "{2}"'.format(prog, fileScsv, fileStxt)
    Popen(cmd, shell = True, stdout = PIPE).communicate()


    plt.boxplot(data)
    plt.show()

    lab = np.transpose(labels)[0]   # FIX ME
    test(data, lab, fileStxt, base,classifier)


# SIFT BoW
from os.path import exists, isdir, basename, join, splitext
import sift
#from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
PRE_ALLOCATION_BUFFER = 1000  # for sift
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
CODEBOOK_FILE = 'codebook.file'
DATASETPATH = 'dataset/'
SIZE_LOCAL_FEATURE = 128 # 64: SURF, 128: SIFT, DF*2: MF
from cPickle import dump, HIGHEST_PROTOCOL, load
from mahotas.features import surf
#from singularityBoF import featuresMF

# only one file
def extractSift(filename):
    features_fname = filename + '.sift'
    sift.process_image(filename, features_fname)
    locs, descriptors = sift.read_features_from_file(features_fname)
    return descriptors

# perform a multifractal spectrum in the first M points detected by sift
def extractMF(filename):
    features_fname = filename + '.sift'
    sift.process_image(filename, features_fname)
    locs, descriptors = sift.read_features_from_file(features_fname)
    sh = min(locs.shape[0], 1000)
    res = np.zeros((sh,SIZE_LOCAL_FEATURE)).astype(np.float32)
    extra = [20,False,True,False,0,0,0]
    WIN = 5
    for i in range(sh):
        x = np.int32(round(locs[i][0]))
        y = np.int32(round(locs[i][1]))
        I = Image.open(filename)
        Nx,Ny = I.size
        a = sg.spec(I.crop((max(x-WIN,0),max(y-WIN,0),min(x+WIN,Nx-1),min(y+WIN,Ny-1))),extra)
        res[i] = a
    print res.shape
    return res

from mahotas import surf
def extractSURF(filename):
    import Image
    I = Image.open(filename)
    f = np.array(I.convert('L').getdata()).reshape(I.size[0],I.size[1])
    f = f.astype(np.uint8)
    return surf.surf(f,descriptor_only=True)

def extractSurf(filename):
    features_fname = filename
    descriptors = surf.surf(np.array(Image.open(filename).convert('L')),descriptor_only=True)
    print "Shape: ", descriptors.shape
    return descriptors

def extractLocal(filename,i):
    f = [extractSift, extractSURF, extractMF] #featuresMF]
    return f[i](filename)

def dict2numpy(dict):
    nkeys = len(dict)
    print "100 no? : ", nkeys
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, SIZE_LOCAL_FEATURE))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0] # num of elements of "value"
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, SIZE_LOCAL_FEATURE)) # eliminate 0's rows in "array"
    print "Tamanio del array devuelto por dict2numpy?: ", array.shape
    return array


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words


def localFeatures(subname,alg):
    base = subname+'C.txt'
    fileStxt = '../exps/'+subname+'S.txt'
    fileScsv = '../exps/'+subname+'S.csv'
    fileCtxt = '../exps/'+base
    fileCcsv = '../exps/'+subname+'C.csv'
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
            nonbread[j] = extractLocal(filename,alg)
            j = j+1
        if (j > (cant-1)*2+1):
            break

    for i in range(1,cant):
        filename = '../images/scanner/baguette/baguette{}.tif'.format(i)
        print filename
        baguette[i] = extractLocal(filename,alg)
        filename = '../images/scanner/lactal/lactal{}.tif'.format(i)
        print filename
        lactal[i] = extractLocal(filename,alg)
        filename = '../images/scanner/salvado/salvado{}.tif'.format(i)
        print filename
        salvado[i] = extractLocal(filename,alg)
        filename = '../images/scanner/sandwich/sandwich{}.tif'.format(i)
        print filename
        sandwich[i] = extractLocal(filename,alg)

        filename = '../images/camera/baguette/slicer/b{}.tif'.format(i)
        print filename
        baguetteC[i] = extractLocal(filename,alg)
        filename = '../images/camera/lactal/l{}.tif'.format(i)
        print filename
        lactalC[i] = extractLocal(filename,alg)
        filename = '../images/camera/salvado/s{}.tif'.format(i)
        print filename
        salvadoC[i] = extractLocal(filename,alg)
        filename = '../images/camera/sandwich/s{}.tif'.format(i)
        print filename
        sandwichC[i] = extractLocal(filename,alg)


    # array of SIFT features
    all_featuresS = {}
    all_featuresC = {}

    arrS = baguette[1:]+ baguetteC[1:]+lactal[1:]+lactalC[1:]+salvado[1:]+salvadoC[1:]+sandwich[1:]+sandwichC[1:]+nonbread[0:(2*(cant-1))]
    # to dict
    for d in range(len(arrS)):
        all_featuresS[d] = arrS[d]

    #for d in range(len(arrS)):
    #    all_featuresC[d] = arrC[d]

    all_features_arrayS = dict2numpy(all_featuresS)
    #all_features_arrayC = dict2numpy(all_featuresC)

    print "Kmeans"
    nfeatures = all_features_arrayS.shape[0]
    nclusters = int(sqrt(nfeatures))
    print "nfeatures: ", nfeatures, " , nclusters: ", nclusters

    # the codebook is made with the training SIFT descriptors
    codebook, distortion = vq.kmeans(all_features_arrayS,
                                             nclusters,
                                             thresh=K_THRESH)

    print "Codebook OK"
    all_word_histgramsS = [[0 for j in range(20)] for i in range(len(all_featuresS))]
    for imagefname in all_featuresS:
        word_histgram = computeHistograms(codebook, all_featuresS[imagefname])
        all_word_histgramsS[imagefname] = word_histgram

    return all_word_histgramsS


#def coefCorr():
    #x = columna
    #y = void fraction
    

#main('gch',0,0)
#main('ccv',1,0)
#main('haralick',2,0)
#main('lbp',3,0)
#main('tas',4,0)
#main('zernike',5,0)
#main('surf',1,True, RANDOM_FOREST)
#main('singularity',6,False,SVM)
#main('sift2',0,True,RANDOM_FOREST)
# Multi Fractal Bag of Features
#main('singularityBoF1000_5',2,True,RANDOM_FOREST)

main('mfs20',6,False,SVM)

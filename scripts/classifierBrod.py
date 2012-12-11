import csv
import sys
import os
from subprocess import *
from gch import colorHistogram
#import pyccv
import colortransforms
import numpy as np
from featureTexture import *
import singularityEntropyCL as sg
import mfs
import localmfrac
import time
import cielab
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

SVM = 0
RANDOM_FOREST = 1
cantDF = 16

def callF(filename,which,extra):
    return features(filename,which,3,False,extra)

def features(filename,i,j,combine,extra):
    #farr = [colorHistogram,ccv,haralick,lbp,tas,zernike, singularity.spec]
    #farr = [localmfrac.localMF]
    #farr = [cielab.lab]
    #farr = [lbp]
    farr = [mfs.mfs]
    #if(combine==True):
    #    return hstack((farr[1](filename),farr[2](filename),farr[3](filename),farr[4](filename),farr[5](filename)))
    t =  time.clock()
    # num of FDs , Open Image?,  convert to grayscale?, (cielab) use L,a,b?
    extra = [5,32]
    res = farr[0](filename,extra)
    t =  time.clock() - t
    #print "Time: ", t
    return res


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
    cnn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, warn_on_equidistant=True)

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
    return pred[i](dtrain,labels,fileStxt, base)


def test(dtrain,labels,fileStxt, base):
    testL = classifierPredict(dtrain,labels,fileStxt, base,0)
    testL = classifierPredict(dtrain,labels,fileStxt, base,1)
    testL = classifierPredict(dtrain,labels,fileStxt, base,2)



def main(subname,which,local):
    base = subname+'C.txt'
    fileStxt = '../exps/'+subname+'S.txt'
    fileScsv = '../exps/'+subname+'S.csv'
    #fileCtxt = '../exps/'+base
    #fileCcsv = '../exps/'+subname+'C.csv'
    cant = 25+1
    dDFs  = cantDF*2

    path = '../images/nonbread/brodatz/'
    dirList=sorted(os.listdir(path))
    print len(dirList)
    total = (len(dirList)-1)*(cant-1)
    brodatz = np.zeros((total,dDFs))

    if(local == True):
        data = localFeatures(subname,which)

        prog = './clas3' # convert.c
        cmd = '{0} "{1}" > "{2}"'.format(prog, fileScsv, fileStxt)
        Popen(cmd, shell = True, stdout = PIPE).communicate()	
        labels = [i for i in range(len(data))]
        labels = map(lambda i: i/((cant-1))+1, labels)
        test(data, labels, fileStxt, base)
        return

    else:    # else global
        import Image
        j = 0
        extra = [cantDF,40,1.15]
        cantClases = len(dirList)-1
        win = 128
        for f in range(cantClases):
            filename = path+dirList[f]
            I = Image.open(filename)
            print filename
            # subdivide image in regions
            # each region is a sample of this texture class
            for i in range(5):
                for j in range(5):
                    brodatz[f*25+i*5+j] = callF(I.crop((win*i,win*j,win*(i+1),win*(j+1))),which,extra)
            if (i > cant*2+1):
                break      

    print "2775?: ", len(brodatz)
    with open(fileScsv, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(brodatz)

    prog = './clas3' # convert.c
    cmd = '{0} "{1}" > "{2}"'.format(prog, fileScsv, fileStxt)
    Popen(cmd, shell = True, stdout = PIPE).communicate()
    labels = [i for i in range(len(brodatz))]
    labels = map(lambda i: i/((cant-1))+1, labels)
    print labels
    test(brodatz, labels, fileStxt, base)


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
from singularityBoF import featuresMF

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
    cantF = 128+1
    cant = 25+1
    import Image
    path = '../images/nonbread/brodatz/'
    pathSub = '../images/nonbread/brodatz/sub/'
    dirList=sorted(os.listdir(path))
    cantClases = len(dirList)-1 # -1 because of the "sub" directory
    total = cantClases*(cant-1)
    #brodatz = np.zeros((total,cantF))
    brodatz = [['Df' for j in range(128)] for i in range(total)]
    win = 128
    for f in range(cantClases-110):
        filename = path+dirList[f]
        I = Image.open(filename)
        print filename
        # subdivide image in regions
        # each region is a sample of this texture class
        for i in range(5):
            for j in range(5):
                I2 = I.crop((win*i,win*j,win*(i+1),win*(j+1)))
                file2 = pathSub+dirList[f]+str(i*5+j)+'.gif'
                I2.save(file2)
                brodatz[f*25+i*5+j]  = extractLocal(file2,0)
        if (i > cant*2+1):
            break      

    print "2775?: ", len(brodatz)
    with open(fileScsv, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(brodatz)

    # array of SIFT features
    all_featuresS = {}

    # to dict
    for d in range(len(brodatz)):
        all_featuresS[d] = brodatz[d]

    all_features_arrayS = dict2numpy(all_featuresS)

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


#main('gch',0,0)
#main('ccv',1,0)
#main('haralick',2,0)
#main('lbp',3,0)
#main('tas',4,0)
#main('zernike',5,0)
#main('surf',1,True, RANDOM_FOREST)
#main('singularity',6,False,SVM)
#main('siftBrodatz2',0,True)

#main('singularityBoF1000_5',2,True,RANDOM_FOREST)

main('mfsbrodatz',6,False)

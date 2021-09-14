import os
from mpi4py import MPI
import math
import shutil

def divideList(lst, processNum):
    start = 0
    end = 0
    numLists = math.ceil(len(lst)/processNum)
    lst1 = [None] * processNum

    for i in range(0, processNum-1):
        end = start + numLists
        lst1[i] = lst[start:end]
        start += numLists  
    lst1[processNum-1] = lst[start:len(lst)]

    return lst1


def birdsInCommon(path1, path2):
    allFilesBirds = os.listdir(path1)
    allFilesCal = os.listdir(path2)
    listAll = []

    for f1 in allFilesBirds:
        for f2 in allFilesCal:
            if f1 == f2:
                listAll.append(f1)

    return listAll


def combineFiles(filePath1, filePath2):
    allFiles1 = os.listdir(filePath1)
    allFiles2 = os.listdir(filePath2)
    
    for bird1 in allFiles1:
        for bird2 in allFiles2:
            if bird1 == bird2:
                pathTo2 = filePath2 + "/" + bird2
                pathTo1 = filePath1 + "/" + bird1
                listOfImages2 = os.listdir(pathTo2)

                for imageName in listOfImages2:
                    pathToImage = pathTo2 + "/" + imageName
                    shutil.move(pathToImage, pathTo1 + "/")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    path1 = input("First path: ")
    path2 = input("Second path: ")

else:
    path1 = None
    path2 = None


path1 = comm.bcast(path1, root=0)
path2 = comm.bcast(path2, root=0)

if rank ==0:

    # get list of matching files:
    filePairs = birdsInCommon(path1, path2)

    # divide list of matching files:
    pairsDivided = divideList(filePairs, size)

    #deal with set 0:
    for files in pairsDivided[0]:
        combineFiles(path1, path2)

    # send other sets of files to procs:
    for proc in range(1, size):
        comm.send(pairsDivided[proc], dest=proc)


else:

    pairsDivided = comm.recv(source=0)

    for setFiles in pairsDivided:
        for file in setFiles:
            combineFiles(path1, path2)


# command:
# mpiexec -np 2 python mpiCombine.py

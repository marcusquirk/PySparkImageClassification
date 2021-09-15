# Katie Wilson-Maher

import os
from mpi4py import MPI
import math
import shutil

# Divides given list into a list of lists and returns the new list
# Length of returned list is equal to the given number of processes
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

# Returns a list of bird species names that exist in both datasets
def birdsInCommon(path1, path2):
    allFilesBirds = os.listdir(path1)
    allFilesCal = os.listdir(path2)

    listAll = []
    for f1 in allFilesBirds:
        for f2 in allFilesCal:
            if f1 == f2:
                listAll.append(f1)

    return listAll

# Moves files from filePath2 to filePath1 if the paths have the same name
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
    # Take input for paths to each folder
    path1 = input("First path: ")
    path2 = input("Second path: ")

else:
    path1 = None
    path2 = None

# Send path1 and path2 to all processes
path1 = comm.bcast(path1, root=0)
path2 = comm.bcast(path2, root=0)

if rank ==0:

    # Create a list of folder names where each folder name exists in both paths
    filePairs = birdsInCommon(path1, path2)

    # Divide list of matching folder names into a new list with the size equal to the amount of processes
    pairsDivided = divideList(filePairs, size)

    # For each folder name in the set assigned to process 0, combine folders
    for files in pairsDivided[0]:
        combineFiles(path1, path2)

    # Send other sets of folder names to each other process
    for proc in range(1, size):
        comm.send(pairsDivided[proc], dest=proc)


else:
    # Receive sets of folder names
    pairsDivided = comm.recv(source=0)

    # For each folder name in the set for the current process, combine folders
    for setFiles in pairsDivided:
        for file in setFiles:
            combineFiles(path1, path2)


# command:
# mpiexec -np 4 python mpiCombine.py

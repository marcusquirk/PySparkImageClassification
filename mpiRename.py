import os
import shutil
from mpi4py import MPI
import re
import math


def distList(lst, size):
	newList = [None] * size
	start = 0
	end = 0
	numLists = math.floor(len(lst)/size)
	for i in range(0, size-1):
		end = start + numLists
		newList[i] = lst[start:end]
		start += numLists  
	newList[size-1] = lst[start:len(lst)] 
	return newList


def standardizeNames(folder, name):
	stringName = name.lower()
	newName = re.sub(r'[^A-Za-z]', '', stringName)
	os.rename(folder + "/" + name, folder + "/" +  str(newName))


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

if rank == 0:

	paths = [path1, path2]

	allFilesPaths = []
	for path in paths:
		allFilesPaths.append(os.listdir(path))

	listFolders = []
	for file in allFilesPaths[0]:
		listFolders.append([file, paths[0]])
	for file in allFilesPaths[1]:
		listFolders.append([file, paths[1]])

	distPaths = distList(listFolders, 4)

	#deal with set 0:
	for setFolders in distPaths[0]:
		standardizeNames(setFolders[1], setFolders[0])

    #send other sets of files to procs:
	for proc in range(1, size):
		comm.send(distPaths[proc], dest=proc)

else:
	sets = comm.recv(source=0)
	for setOfFolders in sets:
		standardizeNames(setOfFolders[1], setOfFolders[0])


# command:
# mpiexec -np 4 python renameFoldersMpi.py

# Katie Wilson-Maher

import os
import shutil
from mpi4py import MPI
import re
import math

# Distributes items from a given list across a new list of a given size
# Items of the lists are lists and hold both a name and an original file path
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

# Standardizes name of a folder in a given folder and renames them
# Resulting names will only contain lower case letters
def standardizeNames(folder, name):
	stringName = name.lower()
	newName = re.sub(r'[^A-Za-z]', '', stringName)
	os.rename(folder + "/" + name, folder + "/" +  str(newName))


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

if rank == 0:

	paths = [path1, path2]

	# Create a list of lists
	# Each internal list contains the names of all folders in the path
	allFilesPaths = []
	for path in paths:
		allFilesPaths.append(os.listdir(path))

	# Create a list of lists
	# Each internal list contains a folder name and the name of the folder its in (one of the original 2 paths)
	listFolders = []
	for file in allFilesPaths[0]:
		listFolders.append([file, paths[0]])
	for file in allFilesPaths[1]:
		listFolders.append([file, paths[1]])

	# Distribute listFolders items into as many lists as there are processes
	distPaths = distList(listFolders, size)

	# Standardize (rename) folder names in the set of folders assigned to process 0
	for setFolders in distPaths[0]:
		standardizeNames(setFolders[1], setFolders[0])

    # Send set of folders to each process
	for proc in range(1, size):
		comm.send(distPaths[proc], dest=proc)

else:
	# Receive sets of folders from process 0
	sets = comm.recv(source=0)

	# Standardize (rename) folder names in the set of folders received
	for setOfFolders in sets:
		standardizeNames(setOfFolders[1], setOfFolders[0])


# command:
# mpiexec -np 4 python mpiRename.py

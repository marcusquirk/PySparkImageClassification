# Liam McCormick and Erlang Long

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from numpy import frombuffer, uint8
from mpi4py import MPI
import os

# function that converts a file path and binary image data into svm format row with string tag
def ConvertToSVM(row):
    # cnvert binary to int array
    pixelInts = frombuffer(row[0], dtype=uint8)
    convertedData = [str(v+1) + ':' + str(k) for v, k in enumerate(pixelInts)]

    tagStr = row[1].split('/')[-2]
    return (tagStr, convertedData)
    

def __main__():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    outFile = 'out.txt'
    pathToImages = './bird_images/train_resized/'
    # dict mapping bir specie to ints for tag in final file
    bird_species = {specie: index for index, specie in enumerate(os.listdir(pathToImages))}

    if rank == 0:
        spark = SparkSession.builder.config("spark.executor.memory","16g").config("spark.driver.memory","15g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","14g").config("spark.driver.maxResultSize","13g").appName('svmMaker').getOrCreate()
        context = spark.sparkContext

        #read all images into dataframe
        images = spark.read.format("image").option("dropInvalid", True).option("recursiveFileLookup",True).load(pathToImages)
        
        #discard irrelevant clolumns
        images = df.select('image.data', 'image.origin').collect()
    else:
        images = None

    # broadcast images and partition for processing
    images = comm.bcast(images, root = 0)

    partitionSize = len(images)//size
    if rank != size-1:
        images = images[(partitionSize*rank):partitionSize*(rank+1)]
    else:
        images = images[partitionSize*rank:]

    # convert entire list to string of lines
    lines = map(ConvertToSVM, images)

    # open file, write lines, and close file
    try:
        file = open(outFile, 'a')
        for line in lines:
            file.write(str(bird_species[line[0]]) + ' ' + ' '.join(line[1])+'\n')
        file.close()
    except:
        exit(-1)
    exit(0)

if __name__ == '__main__':
    __main__()

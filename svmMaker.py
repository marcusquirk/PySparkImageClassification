from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from numpy import frombuffer, uint8, reshape
from mpi4py import MPI
import os

def ConvertToSVM(row):
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
    bird_species = {specie: index for index, specie in enumerate(os.listdir(pathToImages))}

    if rank == 0:
        spark = SparkSession.builder.config("spark.executor.memory","16g").config("spark.driver.memory","15g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","14g").config("spark.driver.maxResultSize","13g").appName('svmMaker').getOrCreate()
        context = spark.sparkContext

        df = spark.read.format("image").option("dropInvalid", True).option("recursiveFileLookup",True).load(pathToImages)
        images = df.select('image.data', 'image.origin').collect()
    else:
        images = None

    images = comm.bcast(images, root = 0)
    
    partitionSize = len(images)//size

    if rank != size-1:
        images = images[(partitionSize*rank):partitionSize*(rank+1)]
    else:
        images = images[partitionSize*rank:]

    lines = map(ConvertToSVM, images)
    file = open(outFile, 'a')
    for line in lines:
        file.write(str(bird_species[line[0]]) + ' ' + ' '.join(line[1])+'\n')
    file.close()
    print('closed')
    exit(0)

if __name__ == '__main__':
    __main__()

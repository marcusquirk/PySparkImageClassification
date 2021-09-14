from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from numpy import frombuffer, uint8, reshape
from mpi4py import MPI


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
    pathToImages = './images/'

    if rank == 0:
        spark = SparkSession.builder.appName('please work').getOrCreate()
        context = spark.sparkContext

        df = spark.read.format("image").option("dropInvalid", True).load(pathToImages)
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

    with open(outFile, 'a') as file:
        for line in lines:
            print(len(line[1]))
            file.write(str(hash(line[0])) + ' ' + ' '.join(line[1]))

if __name__ == '__main__':
    __main__()

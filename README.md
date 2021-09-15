# PySparkImageClassification
Group project for CSC355 at Cornell College

Dustin Johnson, Erlang Long, Liam McCormick, Marcus Quirk, Katie Wilson-Maher

This is a Big Data Machine Learning project using Spark and MPI. We used it to classify images of birds from a combined dataset of bird species from Kaggle and from Caltech.
The total dataset included over 47,000 images.

## The shit you need to run it

1. MPI
2. Spark
3. Python
4. Birds datasets


## Workflow
Depending on what is already available to the user, some of these steps may not be necessary.

1. Use mpiRename.py to standardise the format of the species names in the Kaggle and Caltech datasets. This code accepts user input for the filepaths.
2. Use mpiCombine.py to merge the two datasets. Pictures in the Caltech dataset that match species in the Kaggle dataset will be merged to Kaggle. This code accepts user input for the filepaths.
3. Use resizeGreyscale.py to resize the images to a standard 200x200, and convert them to greyscale. This preprocessing step prepares the pictures for featurisation. The filepaths are hardcoded.
4. Use svmMaker.py to convert the greyscale images into .txt files of the libsvm format. This featurising step allows the images' data to be read by the Spark Machine Learning program. The filepaths are hardcoded.
5. Use mlr.py to train the Multiclass Logistic Regression model. The input file is hardcoded. This prints some of the evaluation statistics and also saves the model as a .parquet file.
6. Us readModel.py to test the classification model on other preprocessed images. The filepaths are hardcoded.

## Datasets
https://www.kaggle.com/gpiosenka/100-bird-species
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

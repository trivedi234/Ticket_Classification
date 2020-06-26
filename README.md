#Description

This package is to be implemented using amazon-sagemaker notebook instance using pytorch36 compiler.
The core file is the ipython notebook "Ticket Classification.ipynb"
Supporting modules, used by the core file are provided in Train and Serve folders.
The core file creates a data directory outside the current folder, where all the generated data and utility objects are stored.
Also provided is a webpage codebase, which can be updated and used as the main web-app to acces the model implementation.

#Requirements

##Python-Version
3.6

##Python Packages
os
re
gc
pickle
numpy
pandas
nltk
sklearn
collections
boto3
sagemaker

##amazon-sagemaker
###instances
ml.t2.medium (x1)
ml.p2.xlarge(x3)
ml.m4.xlarge(x3)

##amazon API Gateway
##amazon Lambda


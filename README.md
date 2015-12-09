Parallelized Python/Cython implementation of Latent Dirichlet allocation
Final project for CS205 at Harvard University
Written by Charles Liu, Nicolas Drizard, and Virgile Audi

# System Requirements:

This package was tested on OSX. We ran experiments on Python 2.7 with needed packages:

- Numpy
- Threading
- Cython

The execution of the Cython scripts require a C compiler.

# Installation:

To install the package, download the zip folder from the git repository. We are working to have a pip install link soon.

# Documentation:

This Python package can be used to perform efficient topic modeling using Latent Dirichlet Allocation. More details on LDA can be found in the IPython notebook below. 

The organisation of the package is as follow:

 - Two classes: 
    
    * The *oviLDA* class to perform Online Variational Inference and the *cgsLDA* class to perform Collapsed Gibbs Sampling
 
    * These 2 classes have particular constructors based on the specificity of each algorithm but have identical methods:
 
        - A fit method taking as a parameter the corpus we want to fit the LDA on
        
        - A transform method that can apply an existing model to new documents and returns the topic assignments for these new documents
    
 -  Useful functions related to the LDA model in the LDAutil folder:
    
    * A print_topic function that prints the top n words from topics wanted by the user
    
    * A perplexity function that evaluates the fit of the model
    
    * A LogLikelihood function that evaluates the log-likelihood of the model 
    
More details on these functions and what they actually evaluate are present in the Ipython notebook.
    
 - A subset of the Reuters news dataset in the form of a document term matrix and the associated vocabulary.
    

# Test to run:

For you to test if your system is up to the requirements and to showcase the package in action, we included a Python test.py file.

You can run both versions of LDA by commenting and uncommenting respectively lines 36 and 39.

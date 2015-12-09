Parallelized Python/Cython implementation of Latent Dirichlet allocation
Final project for CS205 at Harvard University
Written by Charles Liu, Nicolas Drizard, and Virgile Audi

# System Requirements:

This package was tested on OSX. We ran experiments on Python 2.7 with needed packages:

- Numpy
- Threading

# Installation:

To install the package, download the zip folder from the git repository. We are working to have a pip install link soon.

# Documentation:

This Python package can be used to perform efficient topic modeling using Latent Dirichlet Allocation. More details on LDA can be found in the IPython notebook below. 

The organisation of the package is as follow:

 - Two classes: 
    
    * The "oviLDA" class to perform Online Variational Inference and the "cgsLDA" class to perform Collapsed Gibbs Sampling
 
    * These 2 classes have particular constructors based on the specificity of each algorithm but have identical methods:
 
        - A fit method taking as a parameter the corpus we want to fit the LDA on
        
        - A transform method that can apply an existing model to new documents and returns the topic assignments for these new documents
    
 -  Useful functions related to the LDA model in the LDAutil folder:
    
    * A print_topic function that prints the top n words from topics wanted by the user
    
    * A perplexity function that evaluates the fit of the model
    
    * A LogLikelihood function that evaluates the log-likelihood of the model
    
    
 
 

# Test to run



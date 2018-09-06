# OPTaaS Examples

Two examples optimising an SVM using Mind Foundry's [Optimizer-as-a-Service](https://mindfoundry.ai/optaas/) (OPTaaS)

This uses version 1.2.7 of the MF package

## Manual Parameter Optimization: optaas-svm-manual.py

Optimizes an SVM using OPTaaS naively - this is the way I implemented before I learned about the inbuilt methods in OPTaaS to do this with a lot less code. However, the benefit of this method is that it could apply to non-sklearn classifiers.

`python optaas-svm-manual.py <API_KEY>`

## Automatic Pipeline Optimization: optaas-svm-inbuilt.py

Optimizes an SVM on an sklearn pipeline using in-built OPTaaS functionality - far less code

`python optaas-svm-inbuilt.py <API_KEY>`

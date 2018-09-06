# Optimises an SVM using OPTaaS naively - this is the way I implemented before
# I learned about the inbuilt methods in OPTaaS to do this with a lot less code
# The benefit of this method is that it could apply to non-sklearn classifiers

import sys
import sklearn.datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
from mindfoundry.optaas.client.client import OPTaaSClient
from mindfoundry.optaas.client.parameter import FloatParameter, CategoricalParameter
from mindfoundry.optaas.client.client import Goal

# Parse the command line arguments
if len(sys.argv) != 2:
    print("Specify API KEY")
    exit(1)

optaas_api_key = sys.argv[1]
base_url = "https://workshop.optaas.mindfoundry.ai"

# Generate some random data
train_input, train_output = sklearn.datasets.make_classification(
    n_samples=400,
    n_features=80,
    n_informative=8,
    n_redundant=8)

# Create the OPTaaS client
client = OPTaaSClient(base_url, optaas_api_key)

# Setup the Optimisation Parameters, we are going to optimise C, gamma and the
# kernel. It is likely better to only optimise the hyperparameters and not the
# kernel
parameters = [
    CategoricalParameter('kernel', values=['rbf', 'linear', 'poly', 'sigmoid'],
        default='rbf'),
    FloatParameter('c', minimum=0.001, maximum=100),
    FloatParameter('gamma', minimum=0.001, maximum=5)
]

# Our scoring function: Accuracy of cross validation on the training data with
# an SVM
def evaluate_svm_parameters(c, gamma, kernel):
    clf = svm.SVC(C=c, gamma=gamma, kernel=kernel)
    return np.mean(cross_val_score(clf, train_input, train_output,
        scoring='accuracy'))

# Setup the task we run with OPTaaS - aim for 100% accuracy
task = client.create_task(
    title='SVM Hyperparameter Optimisation (Manual)',
    parameters=parameters,
    goal=Goal.max,
    target_score=100.0
)

# Run the task for a maximum of 50 iterations - either 50 iterations or 100%
# accuracy will need to be reached for the task to terminate
best_result, best_config = task.run(evaluate_svm_parameters, max_iterations=50)

# Print the best accuracy received and the configuration used to reach this
print('Best score: ' + str(best_result.score))
print('Best configuration: ' + str(best_config.values))

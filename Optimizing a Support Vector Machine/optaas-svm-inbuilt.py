# Optimises an SVM using OPTaaS

import sys
import sklearn.datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
from mindfoundry.optaas.client.client import OPTaaSClient
from mindfoundry.optaas.client.client import Goal
from mindfoundry.optaas.client.sklearn_pipelines.estimators.svc import SVC
from mindfoundry.optaas.client.sklearn_pipelines.mixin import OptimizablePipeline

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

# Create a pipeline in which a single SVM will be optimised, this will take care
# of optimising all the parameters for us
optimizable_pipeline = OptimizablePipeline([('classification', SVC())])

# Our scoring function: Accuracy of cross validation on the training data with
# an SVM
def evaluate_svm_parameters(pipeline):
    return np.mean(cross_val_score(pipeline, train_input, train_output,
        scoring='accuracy'))

# Setup the task we run with OPTaaS - aim for 100% accuracy
task = client.create_sklearn_task(
    title='SVM Hyperparameter Optimisation (In-built)',
    pipeline=optimizable_pipeline,
    goal=Goal.max,
    target_score=100.0
)

# Run the task for a maximum of 50 iterations - either 50 iterations or 100%
# accuracy will need to be reached for the task to terminate
best_result, best_config = task.run(evaluate_svm_parameters, max_iterations=50)

# Print the best accuracy received and the configuration used to reach this
print('Best score: ' + str(best_result.score))
print('Best configuration: ' + str(best_config.values))

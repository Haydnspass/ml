# coding: utf-8

# Structured Learning:
# =====================
# In this exercise we will implement a structured learning system for
# foreground background segmentation.
# We will learn the weights of a CRF Potts model.
# 
# 
# The first step is to  import all needed modules

# In[1]:

# misc
import numpy
import sys

# visualization
import matplotlib.pyplot as plt
import pylab

# features
import skimage.filters

# discrete graphical model package
from dgm.models import *
from dgm.solvers import *
from dgm.value_tables import *
# misc. tools
from tools import make_toy_dataset, norm01

import matplotlib.pyplot as plt

from tools import make_toy_dataset, norm01
from skimage.morphology import disk

from skimage import feature as skFeature

#parallel execution
from joblib import Parallel, delayed
import multiprocessing


def getNormalizedPicture(raw):
    m = 1 / (numpy.max(raw) - numpy.min(raw))
    d = -1 * numpy.min(raw) / (numpy.max(raw) - numpy.min(raw))
    normalizedRaw = raw * m + d
    return normalizedRaw


def get_unary_features(raw):
    features = []

    ############################################
    # ADD YOUR CODE HERE
    ############################################
    for i in range(1, 5):
        normalizedRaw = getNormalizedPicture(raw)
        features.append(skimage.filters.gaussian(normalizedRaw, i)[:, :, None])

    features.append(numpy.ones(raw.shape)[:, :, None])
    return numpy.concatenate(features, axis=2)

def get_potts_features(raw):
    features = []

    ############################################
    # ADD YOUR CODE HERE
    ############################################
    for i in range(1, 5):
        normalizedRaw = getNormalizedPicture(raw)
        expImage = skFeature.canny(normalizedRaw, i)[:, :, None]
        # expImage = numpy.exp(skFeature.canny(normalizedRaw, i)[:,:,None])
        features.append(expImage)

    # a constant feature is needed
    features.append(numpy.ones(raw.shape)[:, :, None])
    return numpy.concatenate(features, axis=2)

# Function to set up the weighted Model:

def build_model(raw_data, gt_image, weights):
    shape = raw_data.shape
    n_var = shape[0] * shape[1]
    n_labels = 2
    variable_space = numpy.ones(n_var) * n_labels

    # lets compute some filters for the uanry features
    unary_features = get_unary_features(raw_data)

    # lets compute some filters for the potts features
    potts_features = get_potts_features(raw_data)

    n_weights = potts_features.shape[2] + unary_features.shape[2]

    # print("n_weights",n_weights)
    assert n_weights == len(weights)

    # both graphical models
    gm = WeightedDiscreteGraphicalModel(variable_space=variable_space, weights=weights)
    loss_augmented_gm = WeightedDiscreteGraphicalModel(variable_space=variable_space, weights=weights)

    # convert coordinates to scalar
    def vi(x0, x1):
        return x1 + x0 * shape[1]

    # weight ids for the unaries
    # (just plain numbers to remeber which weights
    # are associated with the unary features)
    weight_ids = numpy.arange(unary_features.shape[2])
    for x0 in range(shape[0]):
        for x1 in range(shape[1]):

            pixel_val = raw_data[x0, x1]
            gt_label = gt_image[x0, x1]
            features = unary_features[x0, x1, :]

            unary_function = WeightedTwoClassUnary(features=features, weight_ids=weight_ids,
                                                   weights=weights)

            if gt_label == 0:
                loss = numpy.array([0, 1])
            else:
                loss = numpy.array([1, 0])

            loss_augmented_unary_function = WeightedTwoClassUnary(features=features, weight_ids=weight_ids,
                                                                  weights=weights, const_terms=-1.0 * loss)

            variables = vi(x0, x1)
            gm.add_factor(variables=variables, value_table=unary_function)
            loss_augmented_gm.add_factor(variables=variables, value_table=loss_augmented_unary_function)

    # add pairwise factors
    # the weight id's for the pairwise factors

    # average over 2 coordinates to extract
    # extract feature vectors for potts functins
    def get_potts_feature_vec(coord_a, coord_b):

        fa = potts_features[coord_a[0], coord_a[1], :]
        fb = potts_features[coord_b[0], coord_b[1], :]
        return (fa + fb) / 2.0

    # weight ids for the potts functions
    # (just plain numbers to remeber which weights
    # are associated with the potts features)
    weight_ids = numpy.arange(potts_features.shape[2]) + unary_features.shape[2]

    for x0 in range(shape[0]):
        for x1 in range(shape[1]):

            # horizontal edge
            if x0 + 1 < shape[0]:
                variables = [vi(x0, x1), vi(x0 + 1, x1)]
                features = get_potts_feature_vec((x0, x1), (x0 + 1, x1))
                # the weighted potts function
                potts_function = WeightedPottsFunction(shape=[2, 2],
                                                       features=features,
                                                       weight_ids=weight_ids,
                                                       weights=weights)
                # add factors to both models
                gm.add_factor(variables=variables, value_table=potts_function)
                loss_augmented_gm.add_factor(variables=variables, value_table=potts_function)

            # vertical edge
            if x1 + 1 < shape[1]:
                variables = [vi(x0, x1), vi(x0, x1 + 1)]
                features = get_potts_feature_vec((x0, x1), (x0, x1 + 1))
                # the weighted potts function
                potts_function = WeightedPottsFunction(shape=[2, 2],
                                                       features=features,
                                                       weight_ids=weight_ids,
                                                       weights=weights)
                # add factors to both models
                gm.add_factor(variables=variables, value_table=potts_function)
                loss_augmented_gm.add_factor(variables=variables, value_table=potts_function)

    # gm, loss augmented and the loss
    return gm, loss_augmented_gm, HammingLoss(gt_image.ravel())

def subgradient_ssvm(dataset, n_iter=20, learning_rate=1.0, c=0.5, lower_bounds=None, upper_bounds=None,
                     convergence=0.001):
    weights = dataset.weights
    n = len(dataset.models_train)

    if lower_bounds is None:
        lower_bounds = numpy.ones(len(weights)) * -1.0 * float('inf')

    if upper_bounds is None:
        upper_bounds = numpy.ones(len(weights)) * float('inf')

    do_opt = True
    for iteration in range(n_iter):

        effective_learning_rate = learning_rate * float(learning_rate) / (1.0 + iteration)

        # compute gradient
        diff = numpy.zeros(weights.shape)
        for gm, gm_loss_augmented, loss_function in dataset.models_train:
            # update the weights to the current weight vector
            gm.change_weights(weights)
            gm_loss_augmented.change_weights(weights)

            # the gt vector
            y_true = loss_function.y_true

            # optimize loss augmented /
            # find most violated constraint
            graphcut = GraphCut(model=gm_loss_augmented)
            y_hat = graphcut.optimize()

            # compute joint feature vector
            phi_y_hat = gm.phi(y_hat)
            phi_y_true = gm.phi(y_true)

            diff += phi_y_true - phi_y_hat

        new_weights = weights - effective_learning_rate * (c / n) * diff

        # project new weights
        where_to_large = numpy.where(new_weights > upper_bounds)
        new_weights[where_to_large] = upper_bounds[where_to_large]
        where_to_small = numpy.where(new_weights < lower_bounds)
        new_weights[where_to_small] = lower_bounds[where_to_small]

        delta = numpy.abs(new_weights - weights).sum()
        if (delta < convergence):
            print("converged")
            break
        #print('iter', iteration, 'delta', delta, "  ", numpy.round(new_weights, 3))

        weights = new_weights

    return weights

class HammingLoss(object):
    def __init__(self, y_true):
        self.y_true = y_true.copy()

    def __call__(self, y_pred):
        """total loss"""
        return numpy.sum(self.y_true != y_pred)

class Dataset(object):
    def __init__(self, models_train, models_test, weights):
        self.models_train = models_train
        self.models_test = models_test
        self.weights = weights



performanceMeasure = numpy.zeros((6,5))
performanceMeasure[0,:] = [0.1, 0.5, 0.9, 5, 10]



#for n in performanceMeasure[0,:]:
def performanceCheck(n):
    print(n)
    # noise and shape parameter for image creation
    noise = 2.0
    shape = (30, 30)

    # make dataset
    x_train, y_train = make_toy_dataset(shape=shape, n_images=5, noise=noise)
    x_test, y_test = make_toy_dataset(shape=shape, n_images=5, noise=noise)


    # visualize the features for a raw image
    unary_features = get_unary_features(x_train[0])
    n_unary_features = unary_features.shape[2]



    # visualize the features for a raw image
    potts_features = get_potts_features(x_train[0])
    n_potts_features = potts_features.shape[2]
    # for i in range(potts_features.shape[2]):
    #     pylab.imshow(potts_features[:, :, i])
    #     pylab.show()



    # Build the weighted models:
    n_weights = n_unary_features + n_potts_features
    weights = numpy.zeros(n_weights)

    # build the graphical models
    models_train = [build_model(x, y, weights) for x, y in zip(x_train, y_train)]
    models_test = [build_model(x, y, weights) for x, y in zip(x_test, y_test)]


    dset = Dataset(models_train, models_test, weights)

    # Learn The Weights:
    lower_bounds = numpy.ones(len(weights)) * (-1.0 * float('inf'))

    # we want the regularizer 'beta' to be positive
    lower_bounds[n_unary_features:n_unary_features + n_potts_features] = n

    weights = subgradient_ssvm(dset, c=0.5, learning_rate=1.0, lower_bounds=lower_bounds, n_iter=100)

    currentPerformance = numpy.zeros((5,1))

    # Test set performance:
    for i, (gm, _, loss_function) in enumerate(models_test):
        gm.change_weights(weights)

        graphcut = GraphCut(model=gm)
        y_pred = graphcut.optimize()

        prediction_image = y_pred.reshape(shape)

        lossValue = loss_function(prediction_image.ravel())
        currentPerformance[i] = lossValue
    return currentPerformance

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(performanceCheck)(n) for n in performanceMeasure[0,:])

for i in range(performanceMeasure.shape[1]):
    performanceMeasure[1:,i] = numpy.transpose(results[i])

print(performanceMeasure)

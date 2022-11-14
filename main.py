import math
import numpy as np
import numpy.random
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import gumbel_r
from scipy.stats import laplace
from scipy.stats import cauchy
import pandas
import robustsp as rsp

figNum = 0


def loadAndShowData(file_name):
    """
    Function for loading and creating the plots of the input data
    :param file_name: name of the file where the data lies
    :return: x: dataFrame with ordered data
    """
    x = pandas.read_csv(file_name, header=None, names=["col1"])
    x.sort_values(by="col1", inplace=True)

    return x


def calculateWeightsMean(y, mean, std_dev, name):
    """
    Function that normalizes and calculates the weights of the given data
    :param y: inputted not normalized data
    :param mean: for normalizing purpose
    :param std_dev: for normalizing purposes
    :param name: name of the input data (to select specific M-estimator)
    :return: w: ndarray with the weights for each point
    """
    # Normalize data in
    normalizeY = (y - mean) / std_dev

    # Define weight function
    if name == "x1":
        w = norm.pdf(normalizeY, scale=8)
        x = np.linspace(-30, 30, num=1000)
        phi = norm.pdf(x, scale=8) * x
        phi_x = norm.pdf(x, scale=8)
    elif name == "x2":
        w = gumbel_r.pdf(normalizeY, scale=0.15)
        x = np.linspace(-5, 5, num=1000)
        phi = gumbel_r.pdf(x, scale=0.15) * x
        phi_x = gumbel_r.pdf(x, scale=0.15)
    elif name == "x3":
        w = laplace.pdf(normalizeY, scale=4.5)
        x = np.linspace(-50, 50, num=5000)
        phi = laplace.pdf(x, scale=4.5) * x
        phi_x = laplace.pdf(x, scale=4.5)
    elif name == "x4":
        w = cauchy.pdf(normalizeY, scale=4)
        x = np.linspace(-50, 50, num=5000)
        phi = cauchy.pdf(x, scale=4) * x
        phi_x = cauchy.pdf(x, scale=4)
    elif name == "x5":
        w = norm.pdf(normalizeY, loc=-0.75, scale=0.5)*0.5 + norm.pdf(normalizeY, loc=0.4, scale=0.45)*0.5
        x = np.linspace(-50, 50, num=5000)
        phi = (norm.pdf(x, loc=-0.75, scale=0.5)*0.5 + norm.pdf(x, loc=0.4, scale=0.45)*0.5) * x
        phi_x = norm.pdf(x, loc=-0.75, scale=0.5)*0.5 + norm.pdf(x, loc=0.4, scale=0.45)*0.5
    else:
        # Gaussian
        w = norm.pdf(normalizeY)
        x = np.linspace(-10, 10, num=1000)
        phi = norm.pdf(x)*x
        phi_x = norm.pdf(x)

    return w, normalizeY, {"x": x, "phi": phi, "phi_x": phi_x}


def runMLocation(data_in, std_dev, xi, name):
    """
    Function that calculates the M-estimator for the location
    :param data_in: dataFrame with the data whose location estimator is to be calculated
    :param std_dev: standard deviation estimated
    :param xi: threshold for exiting the algorithm
    :param name: name of the input data (to calculate tailored weights)
    :return: final location estimator
    :return: array with the evolution of the estimators
    :return: final set of data normalize
    """
    # Initialize
    y = data_in.to_numpy()
    k = 0
    w = []
    meanEstimators = []
    y_norm = []

    # Calculate first robust mean estimator and first candidate
    meanEstimator = data_in.median().values[0]
    meanEstimatorCandidate = meanEstimator + 2 * xi * std_dev  # Just for entering the loop

    while abs(meanEstimatorCandidate - meanEstimator)/std_dev > xi:
        meanEstimators = np.append(meanEstimators, meanEstimator)
        meanEstimator = meanEstimatorCandidate
        # Calculate the weights
        w, y_norm, param = calculateWeightsMean(y, meanEstimator, std_dev, name)

        # Compute location estimates
        meanEstimatorCandidate = np.sum(np.multiply(w, y))/np.sum(w)

        # Increment iteration index
        k += 1

    meanEstimator = meanEstimatorCandidate

    return {"mean": meanEstimator, "means": meanEstimators, "y": y_norm, "param": param}


def plotAndPrintExercise1(original_data, results):
    """
    Function that plots and prints everything asked for in the exercise 1
    :param original_data: data directly taken from the files
    :param results: estimated data out of the M-estimator
    """
    global figNum

    print("The estimated location for " + n + " is " + str(locationEstimator[n]["mean"]))

    # Show initial data
    plt.figure(figNum)
    figNum += 1
    plt.title("Original data " + n)
    plt.hist(original_data, bins=200, density=True, histtype="stepfilled")

    # Show evolution of the location
    plt.figure(figNum)
    figNum += 1
    plt.title("Evolution of the location estimator " + n)
    plt.plot(results["means"])

    # Show weight and phi
    fig, axs = plt.subplots(1, 2)
    figNum += 1
    axs[0].set_title("Phi " + n)
    axs[0].plot(results["param"]["x"], results["param"]["phi"])
    axs[1].set_title("Phi/x " + " + normalize final data " + n)
    axs[1].plot(results["param"]["x"], results["param"]["phi_x"])

    # Show final normalize distribution of data
    plt.hist(results["y"], bins=200, density=True)


def generateOutlierAndEstimateMean(gaussian_mean, gaussian_var, gaussian_size, min_outlier=0, max_outlier=0):
    """
    Function that takes random samples from a gaussian, adds an outlier, computes M-estimator and returns error.
    :param gaussian_mean: Real location of the gaussian from where the random samples will be taken
    :param gaussian_var: Real variance of the gaussian from where the random samples will be taken
    :param gaussian_size: Number of random samples to be taken from the gaussian
    :param min_outlier: Minimum bound of the outlier
    :param max_outlier: Maximum bound of the outlier
    :return errors: error between the calculated mean and the real one
    :return outliers: array with the outliers used
    """
    # Init vars
    errors = []
    estimateMean1 = 0
    gaussianVector = []

    # Create outliers
    outliers = range(min_outlier, max_outlier + 1)

    for outlier in outliers:
        # Create random gaussian vector
        gaussianVector = np.random.normal(gaussian_mean, math.sqrt(gaussian_var), gaussian_size)

        # Append outlier
        gaussianVectorWithOutlier = np.append(gaussianVector, outlier)

        # Compute the mean estimator via Huber
        c = 3*math.sqrt(gaussian_var)  # If c is lowered, better results are achieved
        estimateMean = mu_hat = rsp.MLocHUB(gaussianVectorWithOutlier, c)

        errors = np.append(errors, estimateMean - gaussian_mean)

    return errors, outliers


def plotAndPrintExercise2(errors, outliers):
    """
    Funtion that plots and prints everything necessary for the task 2
    :param errors: errors between the mean estimated and the real mean
    :param outliers: outliers added to the data that has been estimated
    """
    global figNum

    plt.figure(figNum)
    figNum += 1
    plt.title("Error between real mean and m-estimated mean")
    plt.plot(outliers, errors)


if __name__ == '__main__':

    # Load data
    data = {}
    for i in range(1, 6):
        name = 'x' + str(i)
        data[name] = loadAndShowData("data-assignment2/" + str(name))

    # Compute M-estimator for location
    locationEstimator = {}
    for n in data:
        locationEstimator[n] = runMLocation(data[n], 1, 0.000005, n)
        plotAndPrintExercise1(data[n], locationEstimator[n])

    # Add outlier to data and test robustness
    errors = np.zeros(201)
    N = 100

    for i in range(N):
        error, outliers = generateOutlierAndEstimateMean(1, 2, 100, -100, 100)
        errors = np.add(errors, error)
    errors /= N
    plotAndPrintExercise2(errors, outliers)

    plt.show()

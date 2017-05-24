from joblib import Parallel, delayed
import multiprocessing
import numpy

# what are your inputs, and what operation do you want to

performanceMeasure = numpy.zeros((6,5))
performanceMeasure[0,:] = [1.5, 2.0, 2.5, 3.0, 3.5]

def processInput(n):
    return n * n * numpy.ones((5,1))

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(processInput)(n) for n in performanceMeasure[0,:])

for i in range(performanceMeasure.shape[1]):
    performanceMeasure[1:,i] = numpy.transpose(results[i])

print(5)
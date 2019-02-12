

import matplotlib.pyplot as plt
from mpfit import mpfit
import numpy as np




# input data
x_axis 			= [-5.0, -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3, -4.2, -4.1, -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
spectrum 		= [0.0793, -0.1485, -0.1313, 0.1168, 0.0692, -0.0021, 0.1479, -0.0066, 0.1279, 0.1103, 0.0703, 0.0323, -0.0009, 0.0600, -0.0138, 0.0995, -0.0101, -0.1405, -0.0015, -0.0394, -0.1294, 0.1394, -0.0136, 0.0358, 0.1519, 0.0669, -0.0202, -0.0731, 0.2034, 0.1811, 0.1070, 0.2872, 0.1640, 0.3299, 0.3872, 0.2129, 0.2924, 0.3831, 0.5188, 0.5407, 0.7137, 0.7388, 0.7094, 0.8603, 0.9573, 0.9627, 0.9936, 1.0240, 0.9076, 0.9784, 1.1380, 1.0535, 0.8761, 0.8974, 0.9425, 0.8655, 0.7386, 0.6381, 0.8499, 0.5404, 0.5643, 0.4140, 0.5495, 0.5727, 0.3065, 0.3210, 0.1952, 0.2053, 0.2853, 0.2644, 0.1033, 0.1298, 0.0597, 0.0164, 0.0291, -0.0244, -0.0446, 0.0276, 0.0053, -0.1169, 0.1583, -0.0957, -0.0123, 0.0738, 0.1016, 0.1518, -0.0406, -0.0104, 0.0381, 0.1214, 0.0548, 0.0193, -0.0513, 0.1378, 0.1229, -0.0850, -0.1013, 0.1372, 0.0516, -0.0695, -0.0599]
spectrum_rms 	= 0.3

# initial guesses
centroid1 	= 1.
fwhm1 		= 2.
height1 	= 0.5
guessp 		= [centroid1, fwhm1, height1]

# limits
min_cen_1	= -5.
max_cen_1	= 5.

# apply limits and control step size
parinfo 	= [
		{'parname':'centroid 1', 'step':0.001, 'limited': [1,1], 'limits': [min_cen_1, max_cen_1]},  
		{'parname':'fwhm 1','step':0.001,'limited':[1,1],'limits':[0, 100.]}, 
		{'parname':'height 1','step':1.e-6}]

# set arguments for the function mpfitFunc
fa 			= {'spectrum': spectrum, 'x_axis': x_axis, 'spectrum_rms': spectrum_rms}

# define a function for mpfit to minimise
def mpfitFunc(p, fjac, spectrum, x_axis, spectrum_rms):
	
	status = 0 # Used by mpfit. Must be included
	[centroid, fwhm, height] = p # mpfit generates a set of possible parameters 'p'

	model = Gaussian(centroid, fwhm, height)(x_axis) # these parameters describe a Gaussian

	deviations = (spectrum - model) / spectrum_rms # a spectrum of deviations from the model in terms of the spectrum rms

	return [status, deviations] # mpfit minimises the deviations
def Gaussian(mean = None, fwhm = None, height = None):
	return lambda x: height * np.exp(-4. * np.log(2) * (x - mean)**2 / fwhm**2)

# run mpfit
mp = mpfit(mpfitFunc, guessp, parinfo = parinfo, functkw = fa, maxiter = 10000, quiet = True)

# extraxt parameters from mp object
[centroid_fit, fwhm_fit, height_fit] = mp.params
# generate a model from the fitted parameters
model_fit = Gaussian(centroid_fit, fwhm_fit, height_fit)(x_axis)

print 'Fitted Parameters:'
print '\tCentroid = ' + str(centroid_fit)
print '\tFWHM = ' + str(fwhm_fit)
print '\tHeight = ' + str(height_fit)

# Plot!
plt.figure()
plt.plot(x_axis, spectrum, color = '0.5', label = 'Data')
plt.plot(x_axis, model_fit, color = 'red', label = 'Fit')
plt.legend(loc = 0)
plt.show()
plt.close()











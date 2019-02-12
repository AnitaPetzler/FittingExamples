import corner
import emcee
import math
import matplotlib.pyplot as plt
import numpy as np

# data
x_axis 			= [-5.0, -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3, -4.2, -4.1, -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
spectrum 		= [0.0793, -0.1485, -0.1313, 0.1168, 0.0692, -0.0021, 0.1479, -0.0066, 0.1279, 0.1103, 0.0703, 0.0323, -0.0009, 0.0600, -0.0138, 0.0995, -0.0101, -0.1405, -0.0015, -0.0394, -0.1294, 0.1394, -0.0136, 0.0358, 0.1519, 0.0669, -0.0202, -0.0731, 0.2034, 0.1811, 0.1070, 0.2872, 0.1640, 0.3299, 0.3872, 0.2129, 0.2924, 0.3831, 0.5188, 0.5407, 0.7137, 0.7388, 0.7094, 0.8603, 0.9573, 0.9627, 0.9936, 1.0240, 0.9076, 0.9784, 1.1380, 1.0535, 0.8761, 0.8974, 0.9425, 0.8655, 0.7386, 0.6381, 0.8499, 0.5404, 0.5643, 0.4140, 0.5495, 0.5727, 0.3065, 0.3210, 0.1952, 0.2053, 0.2853, 0.2644, 0.1033, 0.1298, 0.0597, 0.0164, 0.0291, -0.0244, -0.0446, 0.0276, 0.0053, -0.1169, 0.1583, -0.0957, -0.0123, 0.0738, 0.1016, 0.1518, -0.0406, -0.0104, 0.0381, 0.1214, 0.0548, 0.0193, -0.0513, 0.1378, 0.1229, -0.0850, -0.1013, 0.1372, 0.0516, -0.0695, -0.0599]
spectrum_rms 	= 0.3

# set parameters for emcee
num_walkers = 20 # needs to be more then twice the number of dimensions. 100 is way too many
num_dim = 3 # number of dimensions of the model (Gaussians have 3)
burn_iterations = 100 # initial phase
final_iterations = 100 # exploration phase


# plot the data
plt.figure()
plt.plot(x_axis, spectrum, color = 'black', label = 'Data')
plt.legend(loc = 0)
plt.show()
plt.close()


def LogPrior(x, x_axis, spectrum, rms):
	
	log_prior = 1.
	[centroid, fwhm, height] = x
	
	# These are naive priors, we only want to limit parameter space
	# centroid
	x_min = min(x_axis)
	x_max = max(x_axis)
	if centroid <= x_max and centroid >= x_min:
		log_prior += np.log(1/(x_max - x_min)) # thus the prior is normalised to integrate to 1 over parameter space
	else:
		return -np.inf

	# fwhm
	if fwhm > 0. and fwhm < 10.:
		log_prior += np.log(1/10.) # thus the prior is normalised to integrate to 1 over parameter space
	else:
		return -np.inf

	# height
	min_spectrum = min(spectrum)
	max_spectrum = max(spectrum)
	if height <= 5. * abs(max_spectrum) and height >= -5. * abs(min_spectrum): # seems verbose but eliminates the edge case where min is positive
		log_prior += np.log(1/(5. * abs(max_spectrum) + 5. * abs(min_spectrum)))
	else:
		return -np.inf

	return log_prior
def LogLikelihood(x, x_axis, spectrum, rms):
	'''
	emcee supplies the set of parameters 'x', which is a location vector of a walker in parameter space. emcee will move the walker to maximise the value returned from this function.
	'''
	log_prior = LogPrior(x, x_axis, spectrum, rms)

	[centroid, fwhm, height] = x

	if log_prior != -np.inf:
		model = Gaussian(centroid, fwhm, height)(x_axis)
		N = len(spectrum)
		log_likelihood = ((-sum((spectrum - model)**2)) / (2. * rms**2)) - (N * np.log(math.sqrt(2 * math.pi) * rms))
		return log_likelihood + log_prior
	else:
		return -np.inf
def Gaussian(mean = None, fwhm = None, height = None, sigma = None, amp = None):
	'''
	Generates a Gaussian profile with the given parameters.
	'''

	if sigma == None:
		sigma = fwhm / (2. * math.sqrt(2. * np.log(2.)))

	if height == None:
		height = amp / (sigma * math.sqrt(2.* math.pi))

	return lambda x: height * np.exp(-((x - mean)**2.) / (2.*sigma**2.))

# initial guess for each walker in a single 2d array
p0 = [[np.random.uniform(-5, 5), np.random.uniform(0, 2), np.random.uniform(2. * min(spectrum), 2. * max(spectrum))] for x in range(num_walkers)]

# initialising the sampler object
sampler = emcee.EnsembleSampler(num_walkers, num_dim, LogLikelihood, args = [x_axis, spectrum, spectrum_rms])

converged = False


# 'burn-in' run
pos, prob, state = sampler.run_mcmc(p0, burn_iterations)

# plot the walkers in the 'burning-in' stage. All walkers should converge
plt.figure()
plt.subplot(311)
plt.title('Burn-in Phase')
for walker in range(sampler.chain.shape[0]):
	plt.plot(range(sampler.chain.shape[1]), sampler.chain[walker,:,0])
plt.subplot(312)
for walker in range(sampler.chain.shape[0]):
	plt.plot(range(sampler.chain.shape[1]), sampler.chain[walker,:,1])
plt.subplot(313)
for walker in range(sampler.chain.shape[0]):
	plt.plot(range(sampler.chain.shape[1]), sampler.chain[walker,:,2])
plt.show()
plt.close()

# final run, starting all the walkers from their last position in the burn-in stage
sampler.run_mcmc(pos, final_iterations)

# initiating a list for the median values
median_parameters = np.ones(num_dim)

# plot the walkers in the final stage and find the median values of each parameter
plt.figure()
plt.subplot(311)
plt.title('Final Phase')
for walker in range(sampler.chain.shape[0]):
	plt.plot(range(sampler.chain.shape[1]), sampler.chain[walker,:,0])
plt.subplot(312)
for walker in range(sampler.chain.shape[0]):
	plt.plot(range(sampler.chain.shape[1]), sampler.chain[walker,:,1])
plt.subplot(313)
for walker in range(sampler.chain.shape[0]):
	plt.plot(range(sampler.chain.shape[1]), sampler.chain[walker,:,2])
plt.show()
plt.close()


# corner plot
figure = corner.corner(np.transpose([sampler.flatchain[:,0], sampler.flatchain[:,1], sampler.flatchain[:,2]]), show_titles = True, range = (0.9, 0.9, 0.9), labels = ['Centroid', 'FWHM', 'Height'])
plt.show()
plt.close()


# generate a model from the fitted parameters
centroid_fit 	= corner.quantile(sampler.flatchain[:,0], [0.16, 0.50, 0.84])
fwhm_fit 		= corner.quantile(sampler.flatchain[:,1], [0.16, 0.50, 0.84])
height_fit 		= corner.quantile(sampler.flatchain[:,2], [0.16, 0.50, 0.84])
model_fit 		= Gaussian(centroid_fit[1], fwhm_fit[1], height_fit[1])(x_axis)


# print fitted parameters
print 'Fitted Parameters:'
print '\tCentroid = ' + str(centroid_fit[1]) + ' +/- ' + str(abs((centroid_fit[2] - centroid_fit[0]) / 2))
print '\tFWHM = ' + str(fwhm_fit[1]) + ' +/- ' + str(abs((fwhm_fit[2] - fwhm_fit[0]) / 2))
print '\tHeight = ' + str(height_fit[1]) + ' +/- ' + str(abs((height_fit[2] - height_fit[0]) / 2))

# Plot model
plt.figure()
plt.plot(x_axis, spectrum, color = '0.5', label = 'Data')
plt.plot(x_axis, model_fit, color = 'red', label = 'Fit')
plt.legend(loc = 0)
plt.show()
plt.close()


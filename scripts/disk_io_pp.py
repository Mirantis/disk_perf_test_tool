import sys
import collections

import scipy.stats as stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from data_stat import med_dev, round_deviation
from data_stat import read_data_agent_result

data = read_data_agent_result(sys.argv[1])

# for run in data:
#     for name, numbers in run['res'].items():
#         # med, dev = round_deviation(med_dev(numbers['iops']))
#         # print name, med, '~', dev
#         distr = collections.defaultdict(lambda: 0.0)
#         for i in numbers['iops']:
#             distr[i] += 1

#         print name
#         for key, val in sorted(distr.items()):
#             print "    ", key, val
#         print



# # example data
# mu = 100 # mean of distribution
# sigma = 15 # standard deviation of distribution
# x = mu + sigma * np.random.randn(10000)

x = data[0]['res'][sys.argv[2]]['iops']
# mu, sigma = med_dev(x)
# print mu, sigma

# med_sz = 1
# x2 = x[:len(x) // med_sz * med_sz]
# x2 = [sum(vals) / len(vals) for vals in zip(*[x2[i::med_sz]
#                                               for i in range(med_sz)])]

mu, sigma = med_dev(x)
print mu, sigma
print stats.normaltest(x)

num_bins = 20
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# add a 'best fit' line

y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'Histogram of IQ: $\mu={}$, $\sigma={}$'.format(int(mu), int(sigma)))

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()

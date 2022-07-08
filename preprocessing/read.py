import numpy as np

file = np.load('./20words_mean_face.npy')
print(file)
np.savetxt('./20words_mean_face.txt', file)

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

PROB_PIXEL_LOST = 0.8

np.random.seed(1)
# Load the images.
orig_img = Image.open("lena512color.tiff")

# Convert to arrays.
Xorig = np.array(orig_img)
rows, cols, colors = Xorig.shape

# K is 1 if the pixel is known,
# 0 if the pixel was corrupted.
# The K matrix is initialized randomly.
K = np.zeros((rows, cols, colors))
for i in xrange(rows):
    for j in xrange(cols):
        if np.random.random() > PROB_PIXEL_LOST:
            for k in xrange(colors):
                K[i, j, k] = 1

Xcorr = K*Xorig
corr_img = Image.fromarray(np.uint8(Xcorr))

# Display the images.
fig, ax = plt.subplots(1, 2,figsize=(10, 5))
ax[0].imshow(orig_img);
ax[0].set_title("Original Image")
ax[0].axis('off')
ax[1].imshow(corr_img);
ax[1].set_title("Corrupted Image")
ax[1].axis('off');

# Recover the original image using total variation in-painting.
from cvxpy import *
variables = []
constraints = []
for i in xrange(colors):
    X = Variable(rows, cols)
    variables += [X]
    constraints.append(mul_elemwise(K[:, :, i], X - Xcorr[:, :, i]) == 0)

prob = Problem(Minimize(tv(*variables)), constraints)
prob.solve(verbose=True, solver=SCS)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load variable values into a single array.
rec_arr = np.zeros((rows, cols, colors), dtype=np.uint8)
for i in xrange(colors):
    rec_arr[:, :, i] = variables[i].value

fig, ax = plt.subplots(1, 2,figsize=(10, 5))
# Display the in-painted image.
img_rec = Image.fromarray(rec_arr)
ax[0].imshow(img_rec);
ax[0].set_title("In-Painted Image")
ax[0].axis('off')

img_diff = Image.fromarray(np.abs(Xorig - rec_arr))
ax[1].imshow(img_diff);
ax[1].set_title("Difference Image")
ax[1].axis('off');

plt.show()

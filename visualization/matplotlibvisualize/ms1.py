import matplotlib.pyplot as plt
import numpy as np

# array = np.array([])

plt.plot([1,2,3,4],[1,4,9,16], 'ro')
plt.ylabel('some labels')

plt.show()

t = np.arange(0.,5.,0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3,'g^')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = {'one' : np.random.rand(10),
     'two' : np.random.rand(10)}

df = pd.DataFrame(d)

plt.scatter(d, '')

plt.show()

# df.plot(style=['o','rx'])
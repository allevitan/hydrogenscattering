import numpy as np
from matplotlib import pyplot as plt

b0toa0 = 2.35
a1 = 0.00026165
a2 = 0.00027469
b1 = 0.00109875


Preal = np.random.normal(0.5,0.18,50000)
Preal[Preal<=0] = 0
Preal[Preal>=1] = 0.999999
Rreal = Preal / (1 - Preal)
Rraw = a1 * Rreal / (a2*Rreal + b1)
Rcalc = Rraw * b0toa0
Pcalc = Rcalc / (1+Rcalc)
real_vals = np.bincount((Preal*100).astype(int))
calc_vals = np.bincount((Pcalc*100).astype(int))

plt.plot(real_vals)
plt.plot(calc_vals)
plt.show()

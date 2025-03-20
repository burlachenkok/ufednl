#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

font_size = 35
plt.rcParams["font.size"] = font_size

np.random.seed(123)

D = 2000
K = 10
items = 20*1000
alpha_all = []

for i in range(items):
  x_arr = [np.random.normal() for i in range(D)]
  x_arr_np = np.array(x_arr)
  x_arr_np /= np.linalg.norm(x_arr_np)

  idx_all = np.argsort(np.abs(x_arr_np))
  top_idx = idx_all[-K:]

  compressed = np.zeros_like(x_arr_np)
  compressed[top_idx] = x_arr_np[top_idx]
  one_nimus_alpha = np.linalg.norm(compressed - x_arr_np)**2 / np.linalg.norm(x_arr_np)**2
  alpha = 1 - one_nimus_alpha
  alpha_all.append(alpha)

alpha_wc = K/D
alpha_avg = np.mean(alpha_all)

print("trials: ", items)
print("a_min: ", min(alpha_all))
print("a_max: ", max(alpha_all))
print("a_avg: ", alpha_avg)
print("a_wc: ", alpha_wc)


plt.hist(alpha_all, bins=50, edgecolor='black', alpha=0.7, density=True, label="$\\alpha = 1 - \\frac{||TopK[K=10](x) - x||^2}{||x||^2}, x \\sim S^{d-1}$")
plt.axvline(alpha_wc, color='red', linestyle='dashed', linewidth=2, label=f'$\\alpha$ (worst case): {alpha_wc:.4f}')
plt.axvline(alpha_avg, color='blue', linestyle='dashed', linewidth=2, label=f'$\\alpha$ (average): {alpha_avg:.4f}')

plt.xlabel('$\\alpha$')
plt.ylabel('Density')
#plt.title(f'Histogram of Contraction Factor $\\alpha$. Dimension $d$={D}.')
plt.legend()
plt.show()

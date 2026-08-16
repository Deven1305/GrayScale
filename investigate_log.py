import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Create output dir
out_dir = "scratch_investigation"
os.makedirs(out_dir, exist_ok=True)

# 1. Theoretical Analysis Setup
x = np.linspace(-0.1, 1.2, 1000)
c_values = [0.01, 0.05, 0.1]
clamp_min = 1e-3

def t_log_clamp(x, c_min=1e-3):
    return np.log(np.maximum(x, c_min))

def t_log_c(x, c):
    return np.log(x + c)

def t_asinh(x, c):
    return np.arcsinh(x / c)

# Derivatives
def dt_log_clamp(x, c_min=1e-3):
    d = 1.0 / np.maximum(x, c_min)
    d[x < c_min] = 0 # Derivative is 0 below clamp
    return d

def dt_log_c(x, c):
    return 1.0 / (x + c)

def dt_asinh(x, c):
    return 1.0 / np.sqrt(x**2 + c**2)

# Plotting the functions
plt.figure(figsize=(12, 8))
plt.plot(x, t_log_clamp(x, clamp_min), label=f'log(clamp(x, {clamp_min}))')
for c in c_values:
    plt.plot(x, t_log_c(x, c), label=f'log(x + {c})')
for c in c_values:
    plt.plot(x, t_asinh(x, c), '--', label=f'asinh(x / {c})')
plt.axvline(0, color='black', linewidth=1, alpha=0.5)
plt.axhline(0, color='black', linewidth=1, alpha=0.5)
plt.title("Transformation Functions")
plt.xlabel("x (signal)")
plt.ylabel("Transformed Value")
plt.legend()
plt.grid(True)
plt.savefig(f"{out_dir}/functions.png")
plt.close()

# Plotting the derivatives (sensitivity to noise)
plt.figure(figsize=(12, 8))
plt.plot(x, dt_log_clamp(x, clamp_min), label=f'd/dx log(clamp(x, {clamp_min}))')
for c in c_values:
    plt.plot(x, dt_log_c(x, c), label=f'd/dx log(x + {c})')
for c in c_values:
    plt.plot(x, dt_asinh(x, c), '--', label=f'd/dx asinh(x / {c})')
plt.axvline(0, color='black', linewidth=1, alpha=0.5)
plt.title("Derivative (Noise Amplification)")
plt.xlabel("x (signal)")
plt.ylabel("Derivative (Multiplier)")
plt.ylim(-2, 120)
plt.legend()
plt.grid(True)
plt.savefig(f"{out_dir}/derivatives.png")
plt.close()


# 2. Empirical Noise Simulation (using exact Degradation setup if available)
# From the degradation replica: y = D_a(x * n + g), n ~ Gamma(L, 1/L), g ~ N(0, sigma^2)
L = 17.7 # median L
sigma = 0.04 # upper end of measured sigma (worst case for log)

# Let's generate a range of dark signals
signals = np.array([0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
N = 100000

print("Empirical Noise Simulation Results:")
print(f"L = {L}, sigma = {sigma}")
print("-" * 80)

# We want to measure the variance stabilization: the variance of the transformed signal
# ideally should be constant across all 'x'.

results = []
for x_val in signals:
    # Simulating x * n + g
    n = np.random.gamma(shape=L, scale=1.0/L, size=N)
    g = np.random.normal(loc=0.0, scale=sigma, size=N)
    
    noisy_x = x_val * n + g
    
    # Calculate variances
    var_raw = np.var(noisy_x)
    
    # log(clamp)
    tx_log_clamp = t_log_clamp(noisy_x, clamp_min)
    var_log_clamp = np.var(tx_log_clamp)
    
    # log(x+c)
    var_log_c = {}
    for c in c_values:
        tx_log_c = t_log_c(noisy_x[noisy_x + c > 0], c) # filter invalid log inputs just for variance calc, though they shouldn't happen much for large c
        var_log_c[c] = np.var(tx_log_c)
        
    # asinh(x/c)
    var_asinh = {}
    for c in c_values:
        tx_asinh = t_asinh(noisy_x, c)
        var_asinh[c] = np.var(tx_asinh)
        
    neg_pct = np.mean(noisy_x < 0) * 100
    clamp_pct = np.mean(noisy_x < clamp_min) * 100
    
    res = {
        'x': x_val,
        'var_raw': var_raw,
        'var_log_clamp': var_log_clamp,
        'var_log_c': var_log_c,
        'var_asinh': var_asinh,
        'neg_pct': neg_pct,
        'clamp_pct': clamp_pct
    }
    results.append(res)
    
    print(f"x = {x_val:.3f} | % < 0: {neg_pct:>5.1f}% | % < {clamp_min}: {clamp_pct:>5.1f}%")
    print(f"  Var(raw):        {var_raw:.6f}")
    print(f"  Var(log_clamp):  {var_log_clamp:.6f}")
    for c in c_values:
        print(f"  Var(log(x+{c})):  {var_log_c.get(c, -1):.6f}")
    for c in c_values:
        print(f"  Var(asinh(x/{c})):{var_asinh[c]:.6f}")
    print()

# 3. Analyze the 623% error claim
# The doc says: sigma=0.04, x=0.05 (dark). RMS error is 1.505 nats, 623% of log-signal spread, 11.9% hit log floor.
# Let's reproduce that specifically
x_val = 0.05
sigma = 0.04
N_claim = 100000
n_claim = np.random.gamma(shape=17.7, scale=1.0/17.7, size=N_claim)
g_claim = np.random.normal(loc=0, scale=sigma, size=N_claim)
noisy_claim = x_val * n_claim + g_claim

# the log floor is the clamp_min (1e-3).
hit_floor = np.mean(noisy_claim < clamp_min) * 100
print(f"Reproduction of claim at x={x_val}, sigma={sigma}:")
print(f"  % samples hitting log floor (< {clamp_min}): {hit_floor:.1f}% (Claimed: 11.9%)")
# The error in nats might be std of the log transformed values, or error from true log(x).
log_x_true = np.log(x_val)
log_x_noisy = t_log_clamp(noisy_claim, clamp_min)
rms_error = np.sqrt(np.mean((log_x_noisy - log_x_true)**2))
print(f"  RMS error in nats: {rms_error:.3f} (Claimed: 1.505)")

# 4. Variance Stabilization across signal range (Plot)
xs = [r['x'] for r in results]
plt.figure(figsize=(10, 6))
plt.plot(xs, [r['var_log_clamp'] for r in results], marker='o', label=f'log(clamp(x, {clamp_min}))')
for c in c_values:
    plt.plot(xs, [r['var_log_c'].get(c, np.nan) for r in results], marker='s', label=f'log(x+{c})')
for c in c_values:
    plt.plot(xs, [r['var_asinh'][c] for r in results], marker='^', linestyle='--', label=f'asinh(x/{c})')
plt.title("Variance Stabilization (Lower and Flatter is Better)")
plt.xlabel("x (signal)")
plt.ylabel("Variance of Transformed Signal")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig(f"{out_dir}/variance_stabilization.png")
plt.close()

print("Analysis complete. Plots saved to scratch_investigation/")

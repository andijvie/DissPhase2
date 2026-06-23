# observations:
# rerunning does not yield significantly different results
# more than 50 active generations is not really necessary
# the number of inactive generations should be around 1000
# no. cores ~ 10?
# CHANGE 6 VARIABLES FOR SCALE (maybe cross sections?)

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.widgets import Slider, Button
from scipy.optimize import root_scalar
from pathlib import Path
from scipy.stats import binom

# all 1000 + 50

Sigma_t = 1.
Sigma_a = 0.01
Sigma_s = Sigma_t-Sigma_a
Sigma_f = Sigma_a
Sigma_c = Sigma_a
A = 1.0
Sbins = 2
aHalf = 25
LHalf = 30
pop = 2000

readFromFile = False
isHomg = False
isFM = True
isConv = False
fmBins = 2
window = 10
write = False

file_path_tallies = Path(r"\\wsl$\Ubuntu\home\andijvie\SCONE\InputFiles\popRed.json")
file_path_source = Path(r"\\wsl$\Ubuntu\home\andijvie\SCONE\InputFiles\Sources")
script_dir = str(Path(__file__).resolve().parent) + "\\data\\"
ext = "_N" + str(pop) + "L" + str(LHalf) + "a" + str(aHalf) + ".npy"
if isHomg:
    script_dir += "homg_"
    ext = "_N" + str(pop) + "L" + str(LHalf) + ".npy"
if isConv:
    ext = "_CONV" + ext
if isFM:
    ext = "_FM" + str(fmBins) + "w" + str(window) + ext
    
with open(file_path_tallies, "r") as datafile:
    data = json.load(datafile)












def SanalyticHom_stable(N, B):
    p = 1 / B
    k = np.arange(1, N + 1)

    # P(K=k), where K ~ Binomial(N, 1/B)
    probs = binom.pmf(k, N, p)

    return np.log2(N) - (B / N) * np.sum(probs * k * np.log2(k))


if readFromFile:
    shannon_entropy = np.load(script_dir + "S" + ext)
else:
    shannon_entropy = np.array(data["inactive"]["shannon_entropy"]["shannonEntropy"])
    if write:
        np.save(script_dir + "S" + ext, shannon_entropy)

generations = np.arange(1, len(shannon_entropy) + 1)

plt.figure(figsize=(16, 9))
plt.plot(generations, shannon_entropy, marker=".", color = 'k')
plt.xlabel("Generation")
plt.ylabel("Shannon entropy")
plt.axhline(SanalyticHom_stable(pop, Sbins), color = 'red', lw = 0.5)
plt.tight_layout()
plt.show()



if readFromFile:
    flux = np.load(script_dir + "F" + ext)
    fluxCycles = flux.shape[0]    
    
else:
    fluxHalf = data["inactive"]["flux"]["Res"]
    fluxCycles = np.array(fluxHalf).shape[0]
    
    # [cycle][bin][0][0=value, 1=std]
    fluxHalf = np.array([
        [entry[0][0] for entry in fluxHalf[cycle]]
        for cycle in range(fluxCycles)
    ])
    
    all_data_source = []
    for i in range(1, fluxCycles + 2):
        filepathSource = os.path.join(file_path_source, f"popRed_source{i}.txt")
        
        with open(filepathSource, "r") as sourcefile:
            values = [float(line.strip()) for line in sourcefile if line.strip()]
        all_data_source.append(values)
        
    fluxNorm = np.array(all_data_source)
    
    fluxCycles = fluxCycles * 2 + 1
    flux = np.empty((fluxCycles, fluxHalf.shape[1]))
    flux[0::2] = fluxNorm
    flux[1::2] = fluxHalf

    if write:
        np.save(script_dir + "F" + ext, flux)

x_centers = np.linspace(-LHalf, LHalf, flux.shape[1], False)
dX = x_centers[1] - x_centers[0]
x_centers += dX / 2

totFlux = np.sum(flux, axis=1) * dX
flux = flux / totFlux[:, None]

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

cycle0 = 0
(line,) = ax.plot(x_centers, flux[cycle0], marker=".", color="k")

ax.set_title(f"Cycle {cycle0}")
ax.set_xlabel("x")
ax.set_ylabel("Normalized flux")
ax.set_xlim((-LHalf, LHalf))
ax.set_ylim((0, 0.1))

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
cycle_slider = Slider(
    ax=ax_slider,
    label="Cycle",
    valmin=0,
    valmax= (fluxCycles - 1) / 2,
    valinit=cycle0,
    valstep=0.5
)

# y-limit slider
ax_ylim = plt.axes([0.2, 0.05, 0.6, 0.03])

ylim_slider = Slider(
    ax=ax_ylim,
    label="Y max",
    valmin=0.005,
    valmax=0.3,
    valinit=0.1,
    valstep=0.001
)

ax_prev = plt.axes([0.05, 0.1, 0.08, 0.04])
btn_prev = Button(ax_prev, "◀")
ax_next = plt.axes([0.85, 0.1, 0.08, 0.04])
btn_next = Button(ax_next, "▶")

def prev_cycle(event):
    cycle = np.round(cycle_slider.val, 1)
    if cycle > 0:
        cycle_slider.set_val(cycle - 0.5)

def next_cycle(event):
    cycle = np.round(cycle_slider.val, 1)
    if cycle < (fluxCycles - 1) / 2:
        cycle_slider.set_val(cycle + 0.5)

btn_prev.on_clicked(prev_cycle)
btn_next.on_clicked(next_cycle)

def update(val):
    cycle = np.round(cycle_slider.val, 1)

    line.set_ydata(flux[int(cycle * 2)])

    ax.set_title(f"Cycle {cycle}")

    # update y limit
    ax.set_ylim((0, ylim_slider.val))

    fig.canvas.draw_idle()

cycle_slider.on_changed(update)
ylim_slider.on_changed(update)

if isFM:
    for binX in np.linspace(-LHalf, LHalf, fmBins + 1):
        ax.axvline(binX, linestyle='--', color = 'lightgrey', lw = 1)

if isHomg:
    plt.show()
    exit()




def getMubar():
    return 0
def getD():
    return 1/(3*(Sigma_t - getMubar() * Sigma_s))
def getLD():
    return np.sqrt(getD()/Sigma_c)
alpha = 1/getLD()




k_max = 2          # how far in kappa to search/plot
samples_per_branch = 800  # resolution per tan-branch for plotting + bracketing
eps = 1e-6            # avoid tan singularities


d = aHalf - LHalf
rhs = -alpha * np.tanh(alpha * aHalf)

def f(k):
    # f(k) = k*tan(k*(a-L)) + alpha*tanh(alpha*a)  
    return k * np.tan(k * d) - rhs

absd = abs(d)
if absd == 0:
    raise ValueError("a-L must be nonzero.")

# poles in k>0: k*abs(d) = (m+1/2)pi  ->  k = (m+1/2)pi/abs(d)
m_max = int(np.floor((k_max * absd) / np.pi - 0.5))
poles = (np.arange(m_max + 1) + 0.5) * np.pi / absd
# [0, pole0, pole1, ..., k_max]
bounds = np.concatenate(([0.0], poles[poles < k_max], [k_max]))

roots = []

for i in range(len(bounds) - 1):
    left, right = bounds[i], bounds[i + 1]
    if right - left <= 10 * eps:
        continue

    Lk = left + eps
    Rk = right - eps
    if Lk >= Rk:
        continue

    ks = np.linspace(Lk, Rk, samples_per_branch)
    ys = f(ks)

    finite = np.isfinite(ys)
    ks, ys = ks[finite], ys[finite]
    if len(ks) < 2:
        continue

    s = np.sign(ys)
    # indices where sign changes between consecutive points
    idx = np.where(s[:-1] * s[1:] < 0)[0]

    for j in idx:
        a_k, b_k = ks[j], ks[j + 1]
        try:
            sol = root_scalar(f, bracket=(a_k, b_k), method="brentq", maxiter=200)
            if sol.converged:
                r = sol.root
                # dedupe (can happen if two brackets converge to same root)
                if all(abs(r - rr) > 1e-5 for rr in roots):
                    roots.append(r)
        except ValueError:
            pass
        
        if len(roots) > 0:
            break
    if len(roots) > 0:
        break

roots.sort()

print(f"Found {len(roots)} solution(s) for kappa in (0, {k_max}]:")
for r in roots:
    print(f"  kappa = {r:.10f}")

if False:
    plt.figure(figsize=(7, 3))

    for i in range(len(bounds) - 1):
        left, right = bounds[i], bounds[i + 1]
        Lk = left + eps
        Rk = right - eps
        if Lk >= Rk:
            continue
        ks = np.linspace(Lk, Rk, samples_per_branch)
        ys = f(ks)
        # clip extreme values so the plot stays readable (optional)
        ys = np.clip(ys, -50, 50)
        plt.plot(ks, ys)

    # zero line
    plt.axhline(0, linewidth=1)

    # poles as vertical dashed lines
    for p in poles:
        if 0 < p < k_max:
            plt.axvline(p, linestyle="--", linewidth=0.8, alpha=0.5)

    # roots as markers
    if roots:
        plt.scatter(roots, np.zeros_like(roots), zorder=5)

    plt.xlim(0, k_max)
    plt.ylim(-10, 10)
    plt.xlabel(r"$\kappa$")
    plt.ylabel(r"$f(\kappa) = \kappa \tan(\kappa(a-L)) + \alpha \tanh(\alpha a)$")
    plt.title("Transcendental equation solutions (roots where f=0)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()








kappa = roots[0]
# -----------------------------

# x grid on (-L, L)
x = np.linspace(-LHalf, LHalf, 4000)
dx = x[1] - x[0]

# Precompute constants used in the outer pieces
C_left  = A * np.cosh(alpha * aHalf) / np.cos(kappa * (aHalf - LHalf))   # for -L < x < -a
C_right = A * np.cosh(alpha * aHalf) / np.cos(kappa * (aHalf - LHalf))   # for  a < x <  L

phi = np.empty_like(x)

# Regions (use <= for boundaries to avoid gaps; adjust if you want strict <)
mask_left   = x <= -aHalf
mask_middle = (x > -aHalf) & (x < aHalf)
mask_right  = x >= aHalf

phi[mask_left]   = C_left  * np.cos(kappa * (x[mask_left]  + LHalf))
phi[mask_middle] = A * np.cosh(alpha * x[mask_middle])
phi[mask_right]  = C_right * np.cos(kappa * (x[mask_right] - LHalf))

totFlux = sum(phi) * dx

# Plot
ax.axvline(-aHalf, linestyle='--', color = 'lightgrey', lw = 1)
ax.axvline( aHalf, linestyle='--', color = 'lightgrey', lw = 1)
ax.axvline(-LHalf, linestyle=':', color = 'lightgrey', lw = 1)
ax.axvline( LHalf, linestyle=':', color = 'lightgrey', lw = 1)
ax.plot(x, phi/totFlux, label=r'$\phi(x)$', color = 'grey', linestyle='--', lw=2)

plt.show()


print(f"k/nuBar={1/(1+getD() * kappa**2/Sigma_f)}")



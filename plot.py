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
import matplotlib as mpl
from matplotlib.widgets import Slider, Button
import imageio_ffmpeg
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.optimize import root_scalar
from pathlib import Path
from scipy.stats import binom

mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


# 1,2
# Fission = 55.128750000000004+-0.007887506526938662 %
# Crossing = 46.335950000000004+-0.00775898525726786 %
# Mean = 0.9999163682092556+-1.7332076690763102e-06
# Mean = 0.9999147414486921+-1.714778600837849e-06

# 2,3
# Fission = 41.8809+-0.006002176242757704 %
# Crossing = 28.1067+-0.005643226048706537 %
# Mean = 0.9998986615694166+-2.032761952896332e-06
# Mean = 0.9999006448692153+-1.96800153318021e-06

# 3,4
# Fission = 36.5696+-0.0052229539045256755 %
# Crossing = 18.555850000000003+-0.004565773213194562 %
# Mean = 0.9998784191146883+-2.4989123372072013e-06
# Mean = 0.9998883378269618+-2.21279437669415e-06

# 4,5
# Fission = 34.16475+-0.004916849434488004 %
# Crossing = 12.72145+-0.0037985670361631104 %
# Mean = 0.9998450034205233+-3.0778271505417182e-06
# Mean = 0.9998740855130784+-2.5618799575494207e-06

# 5,6
# Fission = 33.01985+-0.004734643653602666 %
# Crossing = 8.86324+-0.0032156534658137528 %
# Mean = 0.9998072078470825+-4.01732263754316e-06
# Mean = 0.9998676247484909+-2.6581118391990467e-06

# 6,7
# Fission = 32.46360000000001+-0.00471206963289487 %
# Crossing = 6.225935+-0.0027511971474478156 %
# Mean = 0.9997409480885312+-5.05627187804576e-06
# Mean = 0.9998621621730384+-2.7050733546676422e-06

# 7,8
# Fission = 32.1795+-0.004664787285517422 %
# Crossing = 4.390725+-0.0023348319841746644 %
# Mean = 0.9996741543259559+-6.219921452075422e-06
# Mean = 0.9998597971830987+-2.7345772279983945e-06

# 8,9
# Fission = 32.0474+-0.004655414444627889 %
# Crossing = 3.1032900000000003+-0.001961889277405838 %
# Mean = 0.9995466690140845+-9.202098048060388e-06
# Mean = 0.999852284305835+-2.9889159577490597e-06

# 9,10
# Fission = 31.98465+-0.004616488137938296 %
# Crossing = 2.19314+-0.0016668097290332814 %
# Mean = 0.9994666873239436+-1.0317567950720679e-05
# Mean = 0.9998480778672032+-3.1725771048923245e-06

# 10,11
# Fission = 31.9432+-0.0046325022516184484 %
# Crossing = 1.55503+-0.0013999125012835623 %
# Mean = 0.9991463191146882+-1.775185010999582e-05
# Mean = 0.9998441259557345+-3.3595070778416953e-06

# 11,12
# Fission = 31.925700000000003+-0.0045793025548930485 %
# Crossing = 1.1000949999999998+-0.0011975693564883832 %
# Mean = 0.9988653820408161+-2.4353645770929788e-05
# Mean = 0.9998452983673469+-3.5393196339052073e-06

# 12,13
# Fission = 31.9191+-0.004600032409782566 %
# Crossing = 0.7793405+-0.0009911640933392412 %
# Mean = 0.9985934420408163+-2.761545737859557e-05
# Mean = 0.9997672502040816+-4.59253789318886e-06

# 13,14 # wierd peak
# Fission = 31.91+-0.004563014539501272 %
# Crossing = 0.552801+-0.0008475751490723403 %
# Mean = 0.9976630948979592+-4.428162500699443e-05
# Mean = 0.9997232311827959+-8.820424606333221e-06

# 14,15
# Fission = 31.9127+-0.004600185606877727 %
# Crossing = 0.3915885+-0.0007089656043329958 %
# Mean = 0.9973241273469389+-4.7756703876950975e-05
# Mean = 0.9995810775510203+-8.968422274133595e-06

# 15,16
# Fission = 31.912799999999997+-0.004577879160181601 %
# Crossing = 0.2773455+-0.0006031433768012793 %
# Mean = 0.9952693173469388+-9.650162506403167e-05
# Mean = 0.9993415363265306+-1.4444199562967291e-05
#0.9762503553061225+-0.0004287758763210656

# 16,17 # EXTREME peak
# Fission = 31.9071+-0.004589312425609091 %
# Crossing = 0.196184+-0.0005095679696058711 %
# Mean = 0.9925451177551021+-0.0001589449779914078
# Mean = 0.9989599540816326+-4.5122748320401283e-05

# 17,18
# Fission = 31.915899999999997+-0.0045889107393203895 %
# Crossing = 0.139378+-0.0004231201023497938 %
# Mean = 0.9852536426530613+-0.00028944247969936866
# Mean = 0.997124498979592+-5.7864626345396894e-05
#0.9672049165656565+-0.0004130439049063952

# 18,19
# Fission = 31.904350000000004+-0.004627836754945555 %
# Crossing = 0.09807499999999998+-0.00036370555527548654 %
# Mean = 0.9855765114285714+-0.000264284207254934
# Mean = 0.9941657853061224+-0.0001369847365443564

# 19,20 --> n2000 BEST EXAMPLE
# Fission = 31.91455+-0.004655533218453607 %
# Crossing = 0.0701196+-0.0003060274778675928 %
# Mean = 0.985210167755102+-0.00029721374506556616
# Mean = 0.991229318367347+-0.00015555124854630526

# 20,21 # EXTREME peak
# Fission = 31.910899999999998+-0.004649310070322263 %
# Crossing = 0.049047749999999994+-0.0002556855634783669 %
# Mean = 0.9709067781632654+-0.0006156704068996938
# Mean = 0.9878351073118279+-0.0006539701704694061
#0.9098015326262626+-0.0010844866363112675
#0.9665685698989899+-0.00045168028429339794

#0.8451829522828282+-0.00186384885432813
#0.9233020701010102+-0.0009043322789336845

# 21,22
# Fission = 31.90845+-0.00466501221100224 %
# Crossing = 0.0347003+-0.00021385157856151072 %
# Mean = 0.9804505397959182+-0.00035239752449097144
# Mean = 0.970426316122449+-0.0005227764641258203
#0.8412079213131313+-0.001732115433265944
#0.9341322921212121+-0.0008653684720602549

# 22,23
# Fission = 31.916750000000004+-0.004732203797772239 %
# Crossing = 0.02475565+-0.00018243730201702172 %
# Mean = 0.9689574214285713+-0.0004573159540507838
# Mean = 0.9641010432653061+-0.0005120347148167097
#0.8886192916161617+-0.0012078172128686742
#0.8420112656565656+-0.0013417884469294368

# 23,24 
# Fission = 31.91045+-0.004791053921403932 %
# Crossing = 0.0173914+-0.0001558994605763599 %
# Mean = 0.9642361583673468+-0.0004878034121688257
# Mean = 0.8648742373469387+-0.0018016126355499204

# 24,25
# Fission = 31.905649999999998+-0.005092472383081719 %
# Crossing = 0.01218095+-0.00013382182096074617 %
# Mean = 0.9071001918367346+-0.0015332421381408605
# Mean = 0.8347579369387755+-0.002218371601995734


# 25,26
# Fission = 31.908899999999996+-0.004880518557041352 %
# Crossing = 0.008876255000000001+-0.00011548091343702646 %
# Mean = 0.9280483508163265+-0.0011609511816356708
# Mean = 0.7037799934693878+-0.002239030351992416


# 26,27 
# Fission = 31.915150000000004+-0.005170864119390492 %
# Crossing = 0.006289275+-0.00010049511830929898 %
# Mean = 0.8642854420408163+-0.001997259073809611
# Mean = 0.6844953257142857+-0.003235004139862931

# 27,28
# Fission = 31.909699999999997+-0.00485064119440925 %
# Crossing = 0.004518635+-8.040853046319153e-05 %
# Mean = 0.9548350663265306+-0.0008927609629392448
# Mean = 0.5083964616326531+-0.002959580174545978

# 28,29
# Fission = 31.918950000000002+-0.004939831810547804 %
# Crossing = 0.0030920550000000007+-6.620824949439835e-05 %
# Mean = 0.9202269734693879+-0.0009819356832139916
# Mean = 0.412816760244898+-0.004181140202971967



# moving avg entropy increases

#homg200
#2000
# 3.1962394625000004+-0.001580393524076785
# 3.3075596083333334+-0.0003291178182852963
#1000 --> best
# Mean = 3.0339008208333333+-0.003774278620009232
# Mean = 3.2720135666666668+-0.0007231420140006204


Sigma_t = 1.
Sigma_a = 0.01
Sigma_s = Sigma_t-Sigma_a
Sigma_f = Sigma_a
Sigma_c = Sigma_a
A = 1.0
minEntr = 100
maxFlux = 500
Sbins = 2
aHalf = 19
LHalf = 20
pop = 2000

readFromFile = False
isHomg = False
isFM = False
isConv = False
fmBins = 2
window = 200
write = False
animation = False

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






if not isHomg and not readFromFile:
    FM = np.array(data["inactive"]["fm"]["FM"])[:,:,0]
    U_FM = np.array(data["inactive"]["fm"]["FM"])[:,:,1]
    print(FM)
    print("Fission = " + str(100 * (FM[1,1] + FM[0,0])/2) + "+-" + str(100 * np.sqrt(U_FM[1,1]**2 + U_FM[0,0]**2)/2) + " %")
    print("Crossing = " + str(100 * (FM[1,0] + FM[0,1])/2) + "+-" + str(100 * np.sqrt(U_FM[1,0]**2 + U_FM[0,1]**2)/2) + " %")







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


print("Mean = " + str(np.mean(shannon_entropy[minEntr : ])) + "+-" + str(np.std(shannon_entropy)/np.sqrt(len(shannon_entropy[minEntr : ]))))
print("Std [innacurate] = " + str(np.std(shannon_entropy[minEntr : ])))


plt.figure(figsize=(16, 9))
plt.plot(generations, shannon_entropy, marker=".", color = 'k')
plt.xlabel("Generation")
plt.ylabel("Shannon entropy")
if isHomg:
    plt.axhline(SanalyticHom_stable(pop, Sbins), color = 'red', lw = 0.5)
else :
    plt.axhline(1, color = 'red', lw = 0.5)
plt.tight_layout()
plt.show()








if readFromFile:
    flux = np.load(script_dir + "F" + ext)
    fluxCycles = flux.shape[0]    
    
else:
    fluxHalf = data["inactive"]["flux"]["Res"]
    fluxCycles = np.array(fluxHalf).shape[0]
    if fluxCycles > maxFlux:
        fluxCycles = maxFlux
        fluxHalf = fluxHalf[ : maxFlux]
    
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

else:
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




if not animation:
    exit()



flux_whole = flux[0::2]

# Match the number of entropy values and flux snapshots
n_frames = min(len(shannon_entropy), flux_whole.shape[0])

flux_whole = flux_whole[:n_frames]
entropy_anim = shannon_entropy[:n_frames]
gen_anim = np.arange(1, n_frames + 1)

# Optional: if your flux[0] corresponds to generation 0 instead of generation 1,
# use this instead:
# gen_anim = np.arange(0, n_frames)

fig, (ax_entropy, ax_flux) = plt.subplots(
    2,
    1,
    figsize=(16, 9),
    gridspec_kw={"height_ratios": [1, 2]},
    constrained_layout=True
)

# -------------------------
# Top plot: Shannon entropy
# -------------------------
ax_entropy.plot(
    generations,
    shannon_entropy,
    marker=".",
    color="k",
    lw=1
)

if isFM:
    ax_entropy.axvline(
                5 if isHomg else 10,
                linestyle="--",
                color="lightgrey",
                lw=1,
                label = 'First FM cycle'
            )

entropy_dot, = ax_entropy.plot(
    [gen_anim[0]],
    [entropy_anim[0]],
    marker="o",
    color="red",
    markersize=8
)

ax_entropy.set_xlabel("Generation")
ax_entropy.set_ylabel("Shannon entropy")
ax_entropy.set_xlim(generations[0], generations[-1])

ymin = np.nanmin(shannon_entropy)
ymax = np.nanmax(shannon_entropy)
yrange = ymax - ymin if ymax > ymin else 1
ax_entropy.set_ylim(ymin - 0.05 * yrange, ymax + 0.05 * yrange)

if isHomg:
    ax_entropy.axhline(
        SanalyticHom_stable(pop, Sbins),
        color="red",
        lw=0.8,
        linestyle="--",
        label="Analytic homogeneous entropy"
    )
else:
    ax_entropy.axhline(
        1,
        color="red",
        lw=0.8,
        linestyle="--"
    )

ax_entropy.legend(loc="lower right")

# -------------------------
# Bottom plot: flux profile
# -------------------------
if isHomg:
    flux_line, = ax_flux.plot(
        x_centers,
        flux_whole[0],
        marker=".",
        color="k",
        lw=1
    )
else:
    flux_bar = ax_flux.bar(
        x_centers,
        flux_whole[0],
        1,
        lw = 1,
        color = 'grey',
        edgecolor = 'k'
    )

ax_flux.set_xlabel("x")
ax_flux.set_ylabel("Normalized neutron source")
ax_flux.set_xlim((-LHalf, LHalf))

flux_ymax = np.nanmax(flux_whole)
ax_flux.set_ylim(0, 1.05 * flux_ymax)

title = ax_flux.set_title(f"Generation {gen_anim[0]}", y=-0.14)


# Optional FM bin markers
if isFM:
    for binX in np.linspace(-LHalf, LHalf, fmBins + 1):
        ax_flux.axvline(
            binX,
            linestyle="--",
            color="lightgrey",
            lw=1
        )


# If aHalf exists and you want the material/interface markers:
if not isHomg:
    ax_flux.axvline(-aHalf, linestyle="--", color="lightgrey", lw=1)
    ax_flux.axvline( aHalf, linestyle="--", color="lightgrey", lw=1)


def update(frame):
    current_gen = gen_anim[frame]

    if isHomg:
        flux_line.set_ydata(flux_whole[frame])
    else:
        for bar, height in zip(flux_bar, flux_whole[frame]):
            bar.set_height(height)

    entropy_dot.set_data(
        [current_gen],
        [entropy_anim[frame]]
    )

    title.set_text(f"Generation {current_gen}")

    if isHomg:
        return flux_line, entropy_dot, title
    return flux_bar, entropy_dot, title


ani = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=100,
    blit=False
)

ani.save(
    script_dir + "flux_entropy_animation.mp4",
    writer=FFMpegWriter(fps=30, bitrate=3000)
)
plt.show()


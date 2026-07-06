import json
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from pathlib import Path
from scipy.stats import binom

file_path = Path(r"\\wsl$\Ubuntu\home\andijvie\SCONE\InputFiles\popRed.json")
script_dir = str(Path(__file__).resolve().parent) + "\\data\\"

clrs = np.array(["black", "dimgrey", "darkgrey", "black", "dimgrey", "darkgrey", "black", "dimgrey", "darkgrey"])
clrs = np.array(["black", "blue", "red", "lime", "magenta" ,"darkgrey", "black", "blue", "red", "lime","black", "blue", "red", "lime"])
lnstls = np.array(['-','-','-','--','--','--',':',':',':','-.','-.','-.'])
lnstls = np.array(['-','--', ':','-.','-','--', ':','-.','-','--', ':','-.','-.','-','--', ':','-.','-.','-','--', ':','-.','-.','-','--', ':','-.'])


def Sanalytic(N):
    return np.log2(N) - (2.0**(1-N) / N) * np.sum([comb(N, K) * K * np.log2(K) for K in np.arange(N) + 1])

def SanalyticHom(N, B):
    return np.log2(N) - (B**(1-N) / N) * np.sum([
        comb(N, K) * (B - 1) ** (N - K) * K * np.log2(K) 
        for K in np.arange(N) + 1])
    
def SanalyticHom_stable(N, B):
    p = 1 / B
    k = np.arange(1, N + 1)

    # P(K=k), where K ~ Binomial(N, 1/B)
    probs = binom.pmf(k, N, p)

    return np.log2(N) - (B / N) * np.sum(probs * k * np.log2(k))
    



script_dir += "homg_"


# vary fm bins
LHalf = 200
aHalf = 19
#LHalf = 20
pop = 100000
pop = 10000
pop = 1000
#pop = 1000000
fmBinss = np.array([2, 3, 4, 5, 6, 7, 8, 10, 20, 50])
fmBinss = np.array([2, 3, 4, 5, 6, 8, 10, 20, 50])
fmBinss = np.array([8])
fmBinss = np.array([2])
fmBinss = np.array([2, 4, 6, 8])
#fmBinss = np.array([3,4, 5])
#fmBinss = np.array([2, 10, 20, 50])
#fmBinss = np.array([10])
fmBinss = np.array([5, 10])
fmBinss = np.array([10])
Sbins = 10
window = 100
isConv = True

plt.figure(figsize=(16,9))
i = 0
for fmBins in fmBinss:
    ext = "_FM" + str(fmBins) + "w" + str(window) + "_N" + str(pop) + "L" + str(LHalf) + "a" + str(aHalf) + ".npy"
    ext = "_FM" + str(fmBins) + "w" + str(window) + "_N" + str(pop) + "L" + str(LHalf) + ".npy"
    if isConv:
        ext = "_FM" + str(fmBins) + "w" + str(window) + "_CONV_N" + str(pop) + "L" + str(LHalf) + ".npy"

    shannon_entropy = np.load(script_dir + "S" + ext)
    generations = np.arange(1, len(shannon_entropy) + 1)
    plt.plot(generations, shannon_entropy, color = clrs[i], linestyle = lnstls[i], lw = 1, label = r"$bins=$" + str(fmBins), zorder = 0)
    
    i += 1

ext = "_N" + str(pop) + "L" + str(LHalf) + "a" + str(aHalf) + ".npy"
ext = "_N" + str(pop) + "L" + str(LHalf) + ".npy"
if isConv:
    ext = "_CONV_N" + str(pop) + "L" + str(LHalf) + ".npy"

shannon_entropy = np.load(script_dir + "S" + ext)
generations = np.arange(1, len(shannon_entropy) + 1)
plt.plot(generations, shannon_entropy, color = 'dimgrey', alpha = 0.5, linestyle = '-', lw = 2, label = r"No FM", zorder = 10)



#ext = "_FM" + str(10) + "w" + str(window) + "_N" + str(10 * pop) + "L" + str(LHalf) + ".npy"
#
#shannon_entropy = np.load(script_dir + "S" + ext)
#generations = np.arange(1, len(shannon_entropy) + 1)
#plt.plot(generations, shannon_entropy, color = clrs[i], linestyle = lnstls[i], lw = 1, label = r"$bins=$" + str(fmBins), zorder = 0)
    


plt.axhline(SanalyticHom_stable(pop, Sbins), color = 'red', lw = 0.5)
#plt.axhline(1, color = 'red', lw = 0.5)
plt.xlabel("Cycle", fontsize=12)
plt.ylabel(r"Shannon entropy, $\mathcal{S}$", fontsize=12)
plt.legend(fontsize=11.5, loc="lower right")
plt.tight_layout()
plt.show()




# vary fm window
LHalf = 100
pop = 100000
pop = 2000
fmBins = 5
Sbins = 10
windows = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 27, 30, 33, 36, 40, 45, 50])
windows = np.array([1, 2, 5, 10, 20, 30, 40 ,50])
#windows = np.array([9,10,11, 12, 14])
windows = np.array([10, 20, 30, 40, 50, 100, 200, 400, 800])
windows = np.array([10, 20, 30])
windows = np.array([40, 50, 100])
windows = np.array([200, 400, 800])
plt.figure(figsize=(16,9))
i = 0
for window in windows:
    ext = "_FM" + str(fmBins) + "w" + str(window) + "_N" + str(pop) + "L" + str(LHalf) + ".npy"
    if isConv:
        ext = "_FM" + str(fmBins) + "w" + str(window) + "_CONV_N" + str(pop) + "L" + str(LHalf) + ".npy"

    shannon_entropy = np.load(script_dir + "S" + ext)
    generations = np.arange(1, len(shannon_entropy) + 1)
    plt.plot(generations, shannon_entropy, color = clrs[i], linestyle = lnstls[i], lw = 1, label = r"$w=$" + str(window), zorder = 0)
    
    i += 1

ext = "_N" + str(pop) + "L" + str(LHalf) + ".npy"
if isConv:
    ext = "_CONV_N" + str(pop) + "L" + str(LHalf) + ".npy"

shannon_entropy = np.load(script_dir + "S" + ext)
generations = np.arange(1, len(shannon_entropy) + 1)
plt.plot(generations, shannon_entropy, color = 'dimgrey', alpha = 0.5, linestyle = '-', lw = 2, label = r"No FM", zorder = 10)

plt.axhline(SanalyticHom_stable(pop, Sbins), color = 'red', lw = 0.5)
plt.xlabel("Cycle", fontsize=12)
plt.ylabel(r"Shannon entropy, $\mathcal{S}$", fontsize=12)
plt.legend(fontsize=11.5, loc="lower right")
plt.tight_layout()
plt.show()

exit()


# vary pop hom
LHalf = 5
pops = np.array([100, 1000, 10000, 100000])
plt.figure(figsize=(8,6))
i = 0
for pop in pops:
    ext = "_N" + str(pop) + "L" + str(LHalf) + ".npy"

    shannon_entropy = np.load(script_dir + "S" + ext)
    generations = np.arange(1, len(shannon_entropy) + 1)
    plt.plot(generations, shannon_entropy, color = clrs[i], linestyle = lnstls[i], lw = 1, label = r"$N=$" + str(pop))
    
    i += 1

#plt.axhline(Sanalytic(100))
#plt.title("Halfwidth = "+ str(LHalf) + ", a = "+ str(aHalf) + ", vary population")
plt.xlabel("Cycle", fontsize=12)
plt.ylabel(r"Shannon entropy, $\mathcal{S}$", fontsize=12)
plt.legend(fontsize=11.5, loc="lower right")
plt.tight_layout()
plt.show()





# vary a
#aHalfs = [40, 50, 60, 70, 80, 90, 95, 97, 99]
#LHalf = 100
#pop = 2000
#
#plt.figure(figsize=(10,7))
#
#i = 0
#for aHalf in aHalfs:
#    ext = "_N" + str(pop) + "L" + str(LHalf) + "a" + str(aHalf) + ".npy"
#
#    shannon_entropy = np.load(script_dir + "S" + ext)
#    generations = np.arange(1, len(shannon_entropy) + 1)
#    plt.plot(generations, shannon_entropy, color = clrs[i], linestyle = lnstls[i], lw = 1, label = str(aHalf))
#    
#    i += 1
#
#plt.title("Population = "+ str(pop) + ", Halfwidth = "+ str(LHalf) + ", vary a")
#plt.xlabel("Generation")
#plt.ylabel("Shannon entropy")
#plt.legend(loc = 'lower left')
#plt.tight_layout()
#plt.show()

# vary scale
LHalfs = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
aRat = .8
pop = 2000

plt.figure(figsize=(10,7))

i = 0
for LHalf in LHalfs:
    aHalf = int(aRat * LHalf)
    ext = "_N" + str(pop) + "L" + str(LHalf) + "a" + str(aHalf) + ".npy"

    shannon_entropy = np.load(script_dir + "S" + ext)
    generations = np.arange(1, len(shannon_entropy) + 1)
    plt.plot(generations, shannon_entropy, color = clrs[i], linestyle = lnstls[i], lw = 1, label = str(LHalf))
    
    i += 1

plt.title("Population = "+ str(pop) + ", a/L = "+ str(aRat) + ", vary L")
plt.xlabel("Generation")
plt.ylabel("Shannon entropy")
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.show()






# vary a, L=a+5
aHalfs = [5, 10, 20, 50, 100, 200]
LFiss = 5
pop = 10000

#aHalfs = [5, 10, 20, 50, 100, 200]
#LFiss = 5
#pop = 100000

#aHalfs = [5, 10, 20, 50, 100, 200]
#LFiss = 5
#pop = 2000

plt.figure(figsize=(8,5))

i = 0
for aHalf in aHalfs:
    LHalf = aHalf + LFiss
    ext = "_N" + str(pop) + "L" + str(LHalf) + "a" + str(aHalf) + ".npy"

    shannon_entropy = np.load(script_dir + "S" + ext)
    generations = np.arange(1, len(shannon_entropy) + 1)
    plt.plot(generations, shannon_entropy, color = clrs[i], linestyle = lnstls[i], lw = 1, label = r"$a=$"+str(aHalf))
    
    i += 1

#plt.title("Population = "+ str(pop) + ", Fissile = "+ str(LFiss) + ", vary a")
plt.xlabel("Cycle", fontsize=12)
plt.ylabel(r"Shannon entropy, $\mathcal{S}$", fontsize=12)
plt.legend(fontsize=11.5, loc="lower right")
plt.tight_layout()
plt.show()







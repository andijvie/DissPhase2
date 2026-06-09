import json
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from pathlib import Path

file_path = Path(r"\\wsl$\Ubuntu\home\andijvie\SCONE\InputFiles\popRed.json")
script_dir = str(Path(__file__).resolve().parent) + "\\data\\"

clrs = np.array(["black", "dimgrey", "darkgrey", "black", "dimgrey", "darkgrey", "black", "dimgrey", "darkgrey"])
clrs = np.array(["black", "blue", "red", "lime", "magenta" ,"darkgrey", "black", "blue", "red", "lime","black", "blue", "red", "lime"])
lnstls = np.array(['-','-','-','--','--','--',':',':',':','-.','-.','-.'])
lnstls = np.array(['-','--', ':','-.','-','--', ':','-.','-','--', ':','-.'])


def Sanalytic(N):
    return np.log2(N) - (2.0**(1-N) / N) * np.sum([comb(N, K) * K * np.log2(K) for K in np.arange(N) + 1])


# vary pop
aHalf = 25
LHalf = 30
pops = np.array([100, 500, 1000, 5000, 10000, 50000, 100000])
pops = np.array([100, 1000, 10000, 100000, 1000000])

plt.figure(figsize=(8,3))

i = 0
for pop in pops:
    ext = "_N" + str(pop) + "L" + str(LHalf) + "a" + str(aHalf) + ".npy"

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





script_dir += "homg_"

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

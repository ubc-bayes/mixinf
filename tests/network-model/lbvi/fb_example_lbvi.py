import numpy as np
#from kernel import kernel_sampler, rej_beta
from kernel import *
import pickle as pk
import bokeh.plotting as bkp
import bokeh.palettes as bkpl
from bokeh.plotting import output_file, save
import pandas as pd
import os


# load the network
df = pd.read_csv('../fb2.txt', delim_whitespace=True, )

# convert the datetime to integers
df['rnd'] = pd.to_datetime(df['date']).astype(int)/(1800*10**9) # this converts to 1/2 hour intervals
df['rnd'] = (df['rnd'] - df['rnd'][0]).astype(int) #this shifts the intervals by the first time

# remove any self loop entries
df = df[df.v1 != df.v2]

# shift the indices down by 1 (start at 0)
df.v1 -= 1
df.v2 -= 1

# convert the relevant columns to numpy arrays
v1 = np.array(df.v1)
v2 = np.array(df.v2)
rnds = np.array(df.rnd)

# threshold the rounds

rnd_begin = 3950
rnd_end = 10377

v1 = v1[(rnds >= rnd_begin) & (rnds < rnd_end)]
v2 = v2[(rnds >= rnd_begin) & (rnds < rnd_end)]
rnds = rnds[(rnds >= rnd_begin) & (rnds < rnd_end)]


# collect the number of edges and vertices binned into days
# censor to the range when the first edge appears, and before a shift in rate happens
nE = [0] # because we use nE[-1] below; we will remove this entry right after
nV = []
nN = []
V = set()
for n in range(rnd_begin, rnd_end):
    idcs = np.where(rnds == n)[0]
    nE.append(nE[-1] + idcs.shape[0])
    for i in idcs:
        V.add(v1[i])
        V.add(v2[i])
    nV.append(len(V))
    nN.append(n+1)
nE = np.array(nE)[1:] #remove the extra first entry we put before
nV = np.array(nV)
nN = np.array(nN)

# collect the edges into a sparse representation
N = rnd_end - rnd_begin
K = max(v1.max(), v2.max())+1
X = np.zeros((K,K))
for i in range(v1.shape[0]):
    X[v1[i], v2[i]] += 1
degrees = (X+X.T).sum(axis=0)
hist, edges = np.histogram(degrees, density=True, bins=50)
edges = edges[1:]

idcs = np.where(X>0)
Obs = np.zeros((3, idcs[0].shape[0]), dtype=np.int64)
Obs[0,:] = idcs[0]
Obs[1,:] = idcs[1]
Obs[2,:] = X[idcs[0], idcs[1]]

###############################################################################
if not os.path.exists('simu_fb_lbvi.pk'):
    np.random.seed(3)
    T = 2000
    #K = 2010
    Alphs, Gams, Lambs, Ths = kernel_sampler(T, alph = 0.01, gam = 2., lamb = 20., Th = None)
    simu_fb = {'Alph': Alphs, 'Gam': Gams, 'Lamb': Lambs, 'Th': Ths}
    f = open('simu_fb.pk', 'wb')
    pk.dump(simu_fb, f)
    f.close()
else:
    f = open('simu_fb.pk', 'rb')
    simu_fb = pk.load(f)
    f.close()

def predict(alphs, gams, lambs, Ths, N, T):
    K = Ths.shape[1]
    nEVs = []
    HEs = []
    for t in range(T):
        if t%5==0: print(t)
        m = np.random.randint(0, Ths.shape[0], 1)

        ######################
        # simulate from hypers
        #alph = alphs[m]
        #gam = gams[m]
        #lamb = lambs[m]
        #Ths[m] = rej_beta(Ths.shape[1], alph, gam, lamb)
        # to simulate from Ths instead, comment the above block out
        ######################

        # generate N rounds of bernoulli at each pair of vertices using a binomial
        W = np.outer(Ths[m], Ths[m])
        W[np.arange(K), np.arange(K)] = 0
        X = np.random.binomial(N, W, size=W.shape)

        # collect nonzero edge vertex pairs
        V1, V2 = np.where(X>0)

        # create binary sequence of edges for each nonzero pair
        nzEs = np.zeros((V1.shape[0], N))
        for i in range(V1.shape[0]):
            N_pair = X[V1[i], V2[i]]
            idcs = np.random.choice(np.arange(N), size=N_pair, replace=False)
            nzEs[i, idcs] = 1
        # compute number of edges in each round, then cumsum
        nE = np.cumsum(nzEs.sum(axis=0))
        # compute number of unique vertices using a set
        nV = np.zeros(N)
        V = set()
        for n in range(N):
            idcs = np.where(nzEs[:, n]>0)[0]
            for i in idcs:
                V.add(V1[i])
                V.add(V2[i])
            nV[n] = len(V)
        nEVs.append(np.vstack((np.log10(nE), np.log10(nV))))

        # compute the degrees of the final network
        degrees = (X+X.T).sum(axis=0)
        #degrees = np.sort(degrees)
        hist, edges = np.histogram(degrees, density=True, bins=50)
        #hist = hist/hist.sum()
        edges = edges[1:]
        HEs.append([edges, np.log10(hist)])

    return nEVs, HEs


pal = bkpl.Category10[10]

# plot alpha/lambda/gamma histograms
for name_label in [('Alph', 'α'), ('Lamb', 'λ'), ('Gam', 'γ')]:
    name, label = name_label

    fig_fb = bkp.figure(plot_width = 1000, plot_height = 1000, x_axis_label=label, y_axis_label='Probability Mass')
    hist_dense, edges_dense = np.histogram(simu_fb[name])
    hist_dense = hist_dense/hist_dense.sum()
    fig_fb.quad(top=hist_dense, bottom=0, left=edges_dense[:-1], right=edges_dense[1:], fill_color=pal[0], line_color="white")
    fig_fb.xaxis.axis_label_text_font_size = "46pt"
    fig_fb.xaxis[0].ticker.desired_num_ticks = 3
    fig_fb.xaxis.major_label_text_font_size = '40pt'
    fig_fb.yaxis.axis_label_text_font_size = "46pt"
    fig_fb.yaxis.major_label_text_font_size = '40pt'
    bkp.show(fig_fb)

###############################################################################
# plot log edges vs log vertices
f = bkp.figure(plot_width = 800, plot_height=800, x_axis_label='Log10(E)', y_axis_label='Log10(V)')
f.line(np.log10(nE), np.log10(nV), line_width=5, color='black', legend_label='Observation')
p = bkp.figure(plot_width = 800, plot_height=800, x_axis_label='Degree', y_axis_label='log Density')
p.line(edges, np.log10(hist), line_width=5, color='black', legend_label='Observation')
T = 15
lEVs, HEs = predict(simu_fb['Alph'], simu_fb['Gam'], simu_fb['Lamb'], simu_fb['Th'], N, T)
for t in range(T):
    f.line(lEVs[t][0], lEVs[t][1], line_width=2, color=pal[0], alpha=0.1, legend_label='Prediction')

for t in range(T):
    p.line(HEs[t][0], HEs[t][1], line_width=2, color=pal[0], alpha=0.1, legend_label='Prediction')

f.legend.location = "top_left"
f.legend.label_text_font_size = '26pt'
f.xaxis.axis_label_text_font_size = "36pt"
f.xaxis.major_label_text_font_size = '30pt'
f.yaxis.axis_label_text_font_size = "36pt"
f.yaxis.major_label_text_font_size = '30pt'
bkp.show(f)

#p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
p.legend.location = "top_right"
p.legend.label_text_font_size = '26pt'
p.xaxis[0].ticker.desired_num_ticks = 5
p.xaxis.axis_label_text_font_size = "36pt"
p.xaxis.major_label_text_font_size = '30pt'
p.yaxis.axis_label_text_font_size = "36pt"
p.yaxis.major_label_text_font_size = '30pt'
bkp.show(p)

## plot edges vs rounds
#f2 = bkp.figure(plot_width = 800, plot_height=800, x_axis_label='N', y_axis_label='E')
#f2.line(nN, nE, line_width=5, color=pal[0], legend='Observation')
#for t in range(T):
#    f2.line(nN[0]+np.arange(lEVs[t].shape[1]), 10**lEVs[t][0, :], color=pal[1], legend='Samples')
#f2.legend.location = "top_left"
#f2.legend.label_text_font_size = '26pt'
#f2.xaxis.axis_label_text_font_size = "36pt"
#f2.xaxis.major_label_text_font_size = '30pt'
#f2.yaxis.axis_label_text_font_size = "36pt"
#f2.yaxis.major_label_text_font_size = '30pt'
#bkp.show(f2)
#
## plot verts vs rounds
#f3 = bkp.figure(plot_width = 800, plot_height=800, x_axis_label='N', y_axis_label='V')
#f3.line(nN, nV, line_width=5, color=pal[0], legend='Observation')
#for t in range(T):
#    f3.line(nN[0]+np.arange(lEVs[t].shape[1]), 10**lEVs[t][1, :], color=pal[1], legend='Samples')
#f3.legend.location = "top_left"
#f3.legend.label_text_font_size = '26pt'
#f3.xaxis.axis_label_text_font_size = "36pt"
#f3.xaxis.major_label_text_font_size = '30pt'
#f3.yaxis.axis_label_text_font_size = "36pt"
#f3.yaxis.major_label_text_font_size = '30pt'
#bkp.show(f3)

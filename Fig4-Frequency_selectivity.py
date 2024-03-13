import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter

def Hopf_dots(mu, w0, alpha, beta, X, Y, Fx, Fy):
    X_dot = mu*X - w0*Y - (alpha*X - beta*Y)*(X**2 + Y**2) + Fx
    Y_dot = mu*Y + w0*X - (alpha*Y + beta*X)*(X**2 + Y**2) + Fy
    return X_dot, Y_dot

def Hopf_RK2(mu, w0, alpha, beta, X, Y, Fx, Fy, dt, D):
    X_noise = ((2*D*dt)**0.5)*random.gauss(0.0, 1.0)
    Y_noise = ((2*D*dt)**0.5)*random.gauss(0.0, 1.0)
    Xk1, Yk1 = Hopf_dots(mu, w0, alpha, beta, X                   , Y                   , Fx, Fy)
    Xk2, Yk2 = Hopf_dots(mu, w0, alpha, beta, X + Xk1*dt + X_noise, Y + Yk1*dt + Y_noise, Fx, Fy)
    new_X = X + (dt/2)*(Xk1 + Xk2) + X_noise
    new_Y = Y + (dt/2)*(Yk1 + Yk2) + Y_noise
    return new_X, new_Y

def HOPF(dt, mu, w0, alpha, beta, D, Forces_x, Forces_y, r0, phi0):
    N = len(Forces_x)
    X = np.zeros(N, dtype=float)
    Y = np.zeros(N, dtype=float)
    X[0] = r0*np.cos(phi0)
    Y[0] = r0*np.sin(phi0)
    for i in range(1, N):
        X[i], Y[i] = Hopf_RK2(mu, w0, alpha, beta, X[i-1], Y[i-1], Forces_x[i-1], Forces_y[i-1], dt, D)
    return X, Y

def PSD(x, framerate):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    xff = (2.0/(N*framerate))*np.abs(xf[0:int(N/2)])**2
    return f, xff



#Q and BANDWIDTH FUNCTIONS

def find_FFT(x, framerate):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    xff = (2.0/N)*np.abs(xf[0:int(N/2)])
    return f, xff

def find_Q(f, xf):
    assert(len(f) == len(xf))
    xf_thresh = np.max(xf)/(2**0.5)
    xf_list, f_list = [], []
    for i in range(len(f)):
        if xf[i] > xf_thresh:
            xf_list.append( xf[i] )
            f_list.append(  f[i]  )
    xf_cut, f_cut = np.array(xf_list), np.array(f_list)
    f0 = np.sum(f_cut*xf_cut)/np.sum(xf_cut)
    Df = np.max(f_cut)-np.min(f_cut)
    if Df == 0:
        Df = f[1]-f[0]
    return f0/Df

def find_Threshold_Bandwidth(f, xf, threshold):   #0.1 threshold.  noise floor about 0.01 for D=0.01.  Omega0=1 means this is fractional bandwidth
    assert (len(f) == len(xf))
    df = f[1]-f[0]
    count = 0
    for i in range(len(xf)):
        if xf[i] > threshold:
            count += 1
    return count*df

def find_Q_and_tBW(dt, mu, W0, alpha, beta, D, num_seg=10, threshold_for_BW=0.1):
    if mu > 0:
        r0, w0 = (mu/alpha)**0.5, W0 + beta*mu/alpha
    else:
        r0, w0, = 0, W0
    N = 100002
    N_cut = 10000
    x, y = HOPF(dt, mu, w0, alpha, beta, D, np.zeros(N*num_seg + N_cut), np.zeros(N*num_seg + N_cut), r0, 2*np.pi*random.random())
    z_mag = np.zeros(int(N/2))
    for i in range(num_seg):
        f, xf = find_FFT(x[N_cut + i*N : N_cut + (i+1)*N], 2*np.pi/dt)
        z_mag += xf/num_seg
    Q, tBW = find_Q(f[:501], z_mag[:501]), find_Threshold_Bandwidth(f[:501], z_mag[:501], threshold_for_BW)  #501 for 0 to 5W0 range with N=100002, W0=1, dt=2pix10^-3
    return Q, tBW







'''

#Finding tuning curves
D = 0.01
N     = 20000  #10k or 20k
num_seg = 100  #100
N_cut = 10000  #10k just added once
N_all = N*num_seg
dt = 2*np.pi*0.001
W0 = 1
alpha = 1
Mus   = np.array([-10, -1, -0.1, 0, 0.1, 1, 10]  )
Betas = np.array([ 10, 5, 2, 0, -2, -5, -10]  )
XF = np.zeros((len(Betas), len(Mus), int(N/2)))
for b in range(len(Betas)):
    print(b)
    for m in range(len(Mus)):
        mu, beta = Mus[m], Betas[b]
        if mu > 0:
            w0 = W0 + beta*mu/alpha
            r0 = (mu/alpha)**0.5
        else:
            w0 = W0
            r0 = 0
        xf_avg = np.zeros(int(N/2))
        x, y = HOPF(dt, mu, w0, alpha, beta, D, np.zeros(N_all+N_cut), np.zeros(N_all+N_cut), r0, 2*np.pi*random.random())
        for i in range(num_seg):
            f, xf = PSD(x[N_cut + N*i : N_cut + N*(i+1)], 2*np.pi/dt)
            xf_avg += xf/num_seg
        XF[b, m, :] = xf_avg
        
'''

#np.save(r'C:\Users\Justin\Desktop\Criticality\Figures\fig6_tuning_curves', [f, XF, Mus, Betas])
f, XF, Mus_grid, Betas_grid = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig4_tuning_curves.npy', allow_pickle="True")




'''
#Finding Heat Maps
D = 0.01
W0, alpha = 1, 1
dt = 2*np.pi*0.001
Mus = np.linspace(-1, 2, 22)  #21
Betas = np.linspace(-25, 25, 21)  #21
Q = np.zeros(( len(Betas), len(Mus) ))
tBW = np.zeros(( len(Betas), len(Mus) ))
for m in range(len(Mus)):
    print(m)
    for b in range(len(Betas)):
        mu =   Mus[m]
        beta = Betas[b]
        Q[b, m], tBW[b, m] = find_Q_and_tBW(dt, mu, W0, alpha, beta, D, num_seg=60, threshold_for_BW=0.1)   #15mins for each seg
plt.figure()
plt.imshow(Q, cmap='bwr', interpolation='nearest', origin='lower', aspect='auto', vmin=0, vmax=30)  #vmin=0
plt.xticks([0, int(len(Mus)/3),   len(Mus)-1],   [str(min(Mus)), '0' ,str(max(Mus))] )
plt.yticks([0, int(len(Betas)/2), len(Betas)-1], [str(min(Betas)), '0' ,str(max(Betas))] )
plt.axvline(x=int(len(Mus)/3), ls='dashed', color='black')
plt.axhline(y=int(len(Betas)/2), ls='dashed', color='black')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\beta$')
plt.colorbar(label='Quality factor')
plt.figure()
plt.imshow(tBW, cmap='bwr', interpolation='nearest', origin='lower', aspect='auto', vmin=0, vmax=0.5)  #vmin=0
plt.xticks([0, int(len(Mus)/2),   len(Mus)-1],   [str(min(Mus)), '0' ,str(max(Mus))] )
plt.yticks([0, int(len(Betas)/2), len(Betas)-1], [str(min(Betas)), '0' ,str(max(Betas))] )
plt.axvline(x=int(len(Mus)/2), ls='dashed', color='black')
plt.axhline(y=int(len(Betas)/2), ls='dashed', color='black')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\beta$')
plt.colorbar(label='Threshold bandwidth')
'''

#np.save(r'C:\Users\Justin\Desktop\fig9_Q_BW_beta25_mu2_60IC', [Q, tBW] )




Qs, TBW = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig4_heatmaps_beta10_mu2_60IC.npy', allow_pickle="True")
#Qs = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v0\z10m10b_Q_21x21_D=0p01_60IC.npy')
#TBW = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v0\z10m10b_TBW_21x21_D=0p01_60IC_0p1_thresh.npy')
Qs = gaussian_filter(Qs, 2)
TBW = gaussian_filter(TBW, 1.5)

fig = plt.figure(figsize=(11, 5))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.3, hspace=1)
outer = gridspec.GridSpec(4, 8, bottom=-5, right=4)
ax2  = plt.subplot2grid((4, 8), (0,4), colspan=2, rowspan=2)
ax3  = plt.subplot2grid((4, 8), (2,4), colspan=2, rowspan=2)
ax4  = plt.subplot2grid((4, 8), (0,7), colspan=1, rowspan=1)
ax5  = plt.subplot2grid((4, 8), (1,7), colspan=1, rowspan=1)
ax6  = plt.subplot2grid((4, 8), (2,7), colspan=1, rowspan=1)
ax7  = plt.subplot2grid((4, 8), (3,7), colspan=1, rowspan=1)

inner = gridspec.GridSpecFromSubplotSpec(7, 7, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
for b in range(len(Betas_grid)):
    for m in range(len(Mus_grid)):
        mu, beta = Mus_grid[m], Betas_grid[b]
        xf_avg = XF[b, m, :]
        ax = plt.Subplot(fig, inner[b, m])
        ax.plot(f, xf_avg/max(xf_avg), color='black')
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        if b == len(Betas_grid)-1:
            ax.set_xlabel(r'$\mu =$' + str(round(mu, 3)), color='blue')
        if m == 0:
            ax.set_ylabel(r'$\beta =$' + str(round(beta, 3)), color='red')
        if b == 0 and m == len(Mus_grid)-1:
            ax.xaxis.set_label_position("top")
            ax.yaxis.set_label_position("right")
            ax.set_xlabel('Frequency')
            #ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_ylabel('PSD', rotation=270, labelpad=10)
        fig.add_subplot(ax)


MUS, BETAS = np.linspace(-1, 2, 22), np.linspace(-10, 10, 21)
HM2 = ax2.imshow(Qs, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=50)  #vmin=0
ax2.set_xticks([0,   7,   14,  len(MUS)-1])
ax2.set_xticklabels([str(int(min(MUS))), '0', '1' ,str(int(max(MUS)))])
ax2.set_yticks([0, int(1*len(BETAS)/4), int(len(BETAS)/2), int(3*len(BETAS)/4), len(BETAS)-1] )
ax2.set_yticklabels([str(int(min(BETAS))), str(int(min(BETAS)/2)), '0', str(int(max(BETAS)/2)), str(int(max(BETAS)))])
ax2.axvline(x=7, ls='dashed', color='black')
ax2.axhline(y=int(len(BETAS)/2), ls='dashed', color='black')
ax2.set_xlabel(r'$\mu$')
ax2.set_ylabel(r'$\beta$')
fig.colorbar(HM2, ax=ax2, label='Quality factor')

HM3 = ax3.imshow(TBW, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=0.4)  #vmin=0
ax3.set_xticks([0,   7,   14,  len(MUS)-1])
ax3.set_xticklabels([str(int(min(MUS))), '0', '1' ,str(int(max(MUS)))])
ax3.set_yticks([0, int(1*len(BETAS)/4), int(len(BETAS)/2), int(3*len(BETAS)/4), len(BETAS)-1] )
ax3.set_yticklabels([str(int(min(BETAS))), str(int(min(BETAS)/2)), '0', str(int(max(BETAS)/2)), str(int(max(BETAS)))])
ax3.axvline(x=7, ls='dashed', color='black')
ax3.axhline(y=int(len(BETAS)/2), ls='dashed', color='black')
ax3.set_xlabel(r'$\mu$')
ax3.set_ylabel(r'$\beta$')
fig.colorbar(HM3, ax=ax3, label=r'Threshold bandwidth')

beta_slice_color1, beta_slice_color2, beta_slice_color3 = 'blue', 'dodgerblue', 'cyan'
beta_slice_mu_index1, beta_slice_mu_index2, beta_slice_mu_index3 = 12, 15, 20
ax2.axvline(x = beta_slice_mu_index1, ls='dotted', color=beta_slice_color1)
ax2.axvline(x = beta_slice_mu_index2, ls='dotted', color=beta_slice_color2)
ax2.axvline(x = beta_slice_mu_index3, ls='dotted', color=beta_slice_color3)
ax4.plot(BETAS, Qs[:, beta_slice_mu_index1], "-", color=beta_slice_color1)
ax4.plot(BETAS, Qs[:, beta_slice_mu_index2], "-", color=beta_slice_color2)
ax4.plot(BETAS, Qs[:, beta_slice_mu_index3], "-", color=beta_slice_color3)
ax4.set_xlim(-10, 10)
ax4.set_xlabel(r'$\beta$')
ax4.set_ylim(0, 65)
ax4.set_ylabel('Q factor')

mu_slice_color = 'magenta'
mu_slice_beta_index = 12
ax2.axhline(y = mu_slice_beta_index, ls='dotted', color=mu_slice_color)
ax5.plot(MUS, Qs[mu_slice_beta_index, :], "-", color=mu_slice_color)
ax5.set_xlim(-1, 2)
ax5.set_xlabel(r'$\mu$')
ax5.set_ylim(0, 50)
ax5.set_ylabel('Q factor')
ax5.set_xticks([-1, 0, 1, 2])

beta_slice_color1, beta_slice_color2, beta_slice_color3 = 'blue', 'dodgerblue', 'cyan'
beta_slice_mu_index1, beta_slice_mu_index2, beta_slice_mu_index3 = 12, 15, 20
ax3.axvline(x = beta_slice_mu_index1, ls='dotted', color=beta_slice_color1)
ax3.axvline(x = beta_slice_mu_index2, ls='dotted', color=beta_slice_color2)
ax3.axvline(x = beta_slice_mu_index3, ls='dotted', color=beta_slice_color3)
ax6.plot(BETAS, TBW[:, beta_slice_mu_index1], "-", color=beta_slice_color1)
ax6.plot(BETAS, TBW[:, beta_slice_mu_index2], "-", color=beta_slice_color2)
ax6.plot(BETAS, TBW[:, beta_slice_mu_index3], "-", color=beta_slice_color3)
ax6.set_xlim(-10, 10)
ax6.set_xlabel(r'$\beta$')
ax6.set_ylim(0, 0.5)
ax6.set_ylabel('Bandwidth')

mu_slice_color = 'magenta'
mu_slice_beta_index = 15
ax3.axhline(y = mu_slice_beta_index, ls='dotted', color=mu_slice_color)
ax7.plot(MUS, TBW[mu_slice_beta_index, :], "-", color=mu_slice_color)
ax7.set_xlim(-1, 2)
ax7.set_xlabel(r'$\mu$')
ax7.set_ylim(0, 0.5)
ax7.set_ylabel('Bandwidth')
ax7.set_xticks([-1, 0, 1, 2])


ax2.text(-3.0, 1.15, "A", transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax2.text(-0.15, 1.15, "B", transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax3.text(-0.15, 1.15, "C", transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
fig.show()


#plt.savefig(r'C:\Users\Justin\Desktop\Fig4.jpeg', dpi=300)



import random
import numpy as np
import matplotlib.pyplot as plt
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


#TE Functions


def LP_filter(x, framerate, cutoff_f):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    for i in range(len(f)):
        if f[i] > cutoff_f:
            xf[i] = 0.0 + 0.0j
            xf[len(xf)-i-1] = 0.0 + 0.0j
    return np.real(np.fft.ifft(xf))

def PSD(x, framerate):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    xff = (2.0/(N*framerate))*np.abs(xf[0:int(N/2)])**2
    return f, xff

def generate_FM_signal(dt, N=1000000, F=0.1, w=1, w_dev=0.2):   #returns Fx, Fy
    Dw = np.random.normal(0, w_dev, N)
    Dw = LP_filter(Dw, 2*np.pi/dt, 1*w)
    Dw *= w_dev/Dw.std()
    phase = np.zeros(N)
    phase[0] = -np.pi/2
    for i in range(1, len(phase)):
        phase[i] = phase[i-1] + (w + Dw[i])*dt
    return F*np.cos(phase), F*np.sin(phase)

def generate_AM_signal(dt, N=1000000, F_avg=0.1, F_dev=0.01, w=1):   #returns Fx, Fy
    DF = np.random.normal(0, F_dev, N)
    DF = LP_filter(DF, 2*np.pi/dt, 0.5*w)  #cut frequencies below characteristic w or 2w
    DF *= F_dev/DF.std()
    tt = np.linspace(0, (N-1)*dt, N)
    Fx = (F_avg + DF)*np.cos(w*tt - np.pi/2)
    Fy = (F_avg + DF)*np.sin(w*tt - np.pi/2)
    return Fx, Fy



'''
#Estimating Biological Noise
N = 100000
DD = 0.005   #analytic shows D= 0.0008 to 0.02 for HB 1-5nm noise with 50nm amp  or 0.0002 to 0.005 if 1-5% amp noise
xt, yt = HOPF(0.001*2*np.pi, 1, 1, 1, 0, DD, np.zeros(N), np.zeros(N), 1, 0)
rt = (xt**2 + yt**2)**0.5
amp = rt[20000:]
print(amp.mean(), amp.std())
'''

#Noise Sweep Data
Ds = np.logspace(-5, 0, 13)
Ds = np.insert(Ds, 0, 0.0)
Chi1 = np.array([21.54, 21.48, 21.39, 21.13, 20.37, 18.08, 14.06, 10.61,  5.66, 3.33,  2.91,  0.64,  2.39,  1.41])
Chi2 = np.array([99.81, 99.8 , 99.37, 98.3 , 95.01, 87.67, 59.46, 35.46,  9.19, 6.13,  1.23,  1.16,  1.76,  1.41])
Te1 = np.array([0.2563, 0.2558, 0.256 , 0.2474, 0.2434, 0.2373, 0.2328, 0.2041, 0.1486, 0.0984, 0.0544, 0.0313, 0.0188, 0.0127])
Te2 = np.array([0.4076, 0.4077, 0.4081, 0.3973, 0.3845, 0.3585, 0.3074, 0.2391, 0.1593, 0.0818, 0.0378, 0.0245, 0.0154, 0.0106])
Te1_rev = np.array([0.0131, 0.0128, 0.0129, 0.0112, 0.0138, 0.0115, 0.012 , 0.013, 0.0175, 0.0183, 0.0179, 0.0156, 0.0157, 0.0132])
Te2_rev = np.array([0.0134, 0.0145, 0.0123, 0.013 , 0.0161, 0.0139, 0.0151, 0.0185, 0.022 , 0.0243, 0.0214, 0.0179, 0.0169, 0.01  ])
Te1_AM = np.array([0.2236, 0.2213, 0.2178, 0.2136, 0.2033, 0.1909, 0.1762, 0.1501, 0.1178, 0.0902, 0.0525, 0.0305, 0.0193, 0.0142])
Te2_AM = np.array([0.2094, 0.2124, 0.2102, 0.2103, 0.2085, 0.2031, 0.192 , 0.1873, 0.1682, 0.1303, 0.0789, 0.0324, 0.0182, 0.0113])
Te1_AM_rev = np.array([0.052 , 0.0518, 0.0521, 0.0515, 0.0511, 0.0508, 0.049 , 0.047, 0.0436, 0.0381, 0.0344, 0.0269, 0.0223, 0.0182])
Te2_AM_rev = np.array([0.0369, 0.0373, 0.0392, 0.0364, 0.0384, 0.0353, 0.0365, 0.0361, 0.0335, 0.0313, 0.0289, 0.0263, 0.0218, 0.0132])


#LOADING TRACE DATA
CHI = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig2_CHI_D=0p001_64IC.npy')
TE, TE_rev       = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig2_TE_FM_D=0p001_beta25_mu2_N=1M.npy')
TE_am, TE_am_rev = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig2_TE_AM_D=0p001_beta25_mu2_N=1M.npy')
CHI = gaussian_filter(CHI, 1)
TE = gaussian_filter(TE, 1)
TE_am = gaussian_filter(TE_am, 1)

'''
fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.05, top=1.0, wspace=1.0, hspace=1.5)


axA  = plt.subplot2grid((6, 24), (0,0), colspan=6, rowspan=1)
axB  = plt.subplot2grid((6, 24), (0,8), colspan=6, rowspan=1)  #traces on top
axC  = plt.subplot2grid((6, 24), (0,16), colspan=6, rowspan=1)

ax1  = plt.subplot2grid((6, 24), (1,0), colspan=6, rowspan=2)
ax2  = plt.subplot2grid((6, 24), (3,0), colspan=5, rowspan=2)
ax3  = plt.subplot2grid((6, 24), (1,8), colspan=6, rowspan=2)
ax4  = plt.subplot2grid((6, 24), (3,8), colspan=5, rowspan=2)
ax5  = plt.subplot2grid((6, 24), (1,16), colspan=6, rowspan=2)
ax6  = plt.subplot2grid((6, 24), (3,16), colspan=5, rowspan=2)

ax7  = plt.subplot2grid((6, 24), (5,0), colspan=3, rowspan=1)   #cross sections
ax8  = plt.subplot2grid((6, 24), (5,4), colspan=3, rowspan=1)
ax9  = plt.subplot2grid((6, 24), (5,8), colspan=3, rowspan=1)
ax10  = plt.subplot2grid((6, 24), (5,12), colspan=3, rowspan=1)
ax11  = plt.subplot2grid((6, 24), (5,16), colspan=3, rowspan=1)
ax12  = plt.subplot2grid((6, 24), (5,20), colspan=3, rowspan=1)
'''




fig = plt.figure(figsize=(10, 7))
plt.subplots_adjust(left=0.08, right=1.05, bottom=0.08, top=1.0, wspace=0.1, hspace=1.2)


axA  = plt.subplot2grid((6, 24), (0,0), colspan=6, rowspan=1)
axB  = plt.subplot2grid((6, 24), (0,8), colspan=6, rowspan=1)  #traces on top
axC  = plt.subplot2grid((6, 24), (0,16), colspan=6, rowspan=1)

ax1  = plt.subplot2grid((6, 24), (1,0), colspan=6, rowspan=2)
ax2  = plt.subplot2grid((6, 24), (3,0), colspan=5, rowspan=2)
ax3  = plt.subplot2grid((6, 24), (1,8), colspan=6, rowspan=2)
ax4  = plt.subplot2grid((6, 24), (3,8), colspan=5, rowspan=2)
ax5  = plt.subplot2grid((6, 24), (1,16), colspan=6, rowspan=2)
ax6  = plt.subplot2grid((6, 24), (3,16), colspan=5, rowspan=2)

ax7  = plt.subplot2grid((6, 24), (5,0), colspan=2, rowspan=1)   #cross sections
ax8  = plt.subplot2grid((6, 24), (5,4), colspan=2, rowspan=1)
ax9  = plt.subplot2grid((6, 24), (5,8), colspan=2, rowspan=1)
ax10  = plt.subplot2grid((6, 24), (5,12), colspan=2, rowspan=1)
ax11  = plt.subplot2grid((6, 24), (5,16), colspan=2, rowspan=1)
ax12  = plt.subplot2grid((6, 24), (5,20), colspan=2, rowspan=1)



#Illustrations
dt = 0.001*2*np.pi
tt = np.linspace(0, 2, 1000)
x1 = 0.1*np.sin(10*2*np.pi*tt)
axA.plot(tt, x1, color='black')
axA.set_ylim(-0.12, 0.5)
axA.axis('off')

np.random.seed(5)
nn = 20000
x2 = generate_FM_signal(dt, N=2*nn, F=1, w=1, w_dev=0.3)[0]
axB.plot(np.linspace(0, 2, len(x2[:15000])), 0.1*x2[:15000], color='black')
axB.set_ylim(-0.12, 0.5)
axB.axis('off')

np.random.seed(3)
x3 = generate_AM_signal(dt, N=2*nn, F_avg=0.75, F_dev=0.2, w=1)[0]
axC.plot(np.linspace(0, 2, len(x3[:15000])), 0.1*x3[:15000], color='black')
axC.set_ylim(-0.12, 0.5)
axC.axis('off')


ax1.plot(Ds[1:], Chi1[1:], "o-", color='black')
ax1.plot(Ds[1:], Chi2[1:], "o-", color='red')
ax1.axhline(y=Chi1[0], ls="dotted", color='black')
ax1.axhline(y=Chi2[0], ls="dotted", color='red')
ax1.set_xscale("log")
ax1.set_xticks([10**(-5), 10**(-4), 10**(-3), 10**(-2), 0.1, 1])
ax1.set_ylim(0, 110)
ax1.axvspan(0.0002, 0.005, alpha=0.2, color='black')
ax1.set_xlabel("Noise strength")
ax1.set_ylabel(r'$\chi(\Omega_0)$')

ax3.plot(Ds[1:], Te1[1:], "o-", color='black')
ax3.plot(Ds[1:], Te2[1:], "o-", color='red')
ax3.plot(Ds[1:], Te1_rev[1:], "o--", color='black', mfc='none')
ax3.plot(Ds[1:], Te2_rev[1:], "o--", color='red', mfc='none')
ax3.axhline(y=Te1[0], ls="dotted", color='black')
ax3.axhline(y=Te2[0], ls="dotted", color='red')
ax3.set_xscale("log")
ax3.set_xticks([10**(-5), 10**(-4), 10**(-3), 10**(-2), 0.1, 1])
ax3.set_ylim(0, 0.5)
ax3.axvspan(0.0002, 0.005, alpha=0.2, color='black')
ax3.set_xlabel("Noise strength")
ax3.set_ylabel("Transfer entropy (FM) (bits)")

ax5.plot(Ds[1:], Te1_AM[1:], "o-", color='black')
ax5.plot(Ds[1:], Te2_AM[1:], "o-", color='red')
ax5.plot(Ds[1:], Te1_AM_rev[1:], "o--", color='black', mfc='none')
ax5.plot(Ds[1:], Te2_AM_rev[1:], "o--", color='red', mfc='none')
ax5.axhline(y=Te1_AM[0], ls="dotted", color='black')
ax5.axhline(y=Te2_AM[0], ls="dotted", color='red')
ax5.set_xscale("log")
ax5.set_xticks([10**(-5), 10**(-4), 10**(-3), 10**(-2), 0.1, 1])
ax5.set_ylim(0, 0.5)
ax5.axvspan(0.0002, 0.005, alpha=0.2, color='black')
ax5.set_xlabel("Noise strength")
ax5.set_ylabel("Transfer entropy (AM) (bits)")


MUS, BETAS, mu0_index = np.linspace(-1, 2, 22), np.linspace(-10, 10, 21), 7

HM2 = ax2.imshow(CHI, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=110)  #vmin=0
ax2.set_xticks([0,   7,   14,  len(MUS)-1])
ax2.set_xticklabels([str(int(min(MUS))), '0', '1' ,str(int(max(MUS)))])
ax2.set_yticks([0, int(1*len(BETAS)/4), int(len(BETAS)/2), int(3*len(BETAS)/4), len(BETAS)-1] )
ax2.set_yticklabels([str(int(min(BETAS))), str(int(min(BETAS)/2)), '0', str(int(max(BETAS)/2)), str(int(max(BETAS)))])
ax2.axvline(x=mu0_index, ls='dashed', color='black')
ax2.axhline(y=int(len(BETAS)/2), ls='dashed', color='black')
ax2.set_xlabel(r'$\mu$')
ax2.set_ylabel(r'$\beta$')
fig.colorbar(HM2, ax=ax2, label=r'$\chi(\Omega_0)$')
beta_slice_color1, beta_slice_color2, beta_slice_color3 = 'blue', 'dodgerblue', 'cyan'
mu_slice_color = 'magenta'
beta_slice_mu_index1, beta_slice_mu_index2, beta_slice_mu_index3 = 11, 14, 21
mu_slice_beta_index = 13
ax2.axvline(x = beta_slice_mu_index1, ls='dotted', color=beta_slice_color1)
ax2.axvline(x = beta_slice_mu_index2, ls='dotted', color=beta_slice_color2)
ax2.axvline(x = beta_slice_mu_index3, ls='dotted', color=beta_slice_color3)
ax2.axhline(y = mu_slice_beta_index, ls='dotted', color=mu_slice_color)
ax7.plot(BETAS, CHI[:, beta_slice_mu_index1], "-", color=beta_slice_color1)
ax7.plot(BETAS, CHI[:, beta_slice_mu_index2], "-", color=beta_slice_color2)
ax7.plot(BETAS, CHI[:, beta_slice_mu_index3], "-", color=beta_slice_color3)
ax8.plot(MUS, CHI[mu_slice_beta_index, :], "-", color=mu_slice_color)
ax7.set_xlabel(r'$\beta$')
ax7.set_ylabel(r'$\chi(\Omega_0)$')
ax7.set_ylim(0, 130)
ax8.set_xlabel(r'$\mu$')
ax8.set_ylabel(r'$\chi(\Omega_0)$')
ax8.set_xlim(-1, 2)
ax8.set_xticks([-1, 0, 1, 2])
ax8.set_ylim(0, 130)


HM4 = ax4.imshow(TE, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto', vmin=0.1, vmax=0.32)  #vmin=0
ax4.set_xticks([0,   7,   14,  len(MUS)-1])
ax4.set_xticklabels([str(int(min(MUS))), '0', '1' ,str(int(max(MUS)))])
#ax4.set_yticks([0, int(1*len(BETAS)/4), int(len(BETAS)/2), int(3*len(BETAS)/4), len(BETAS)-1] )
#ax4.set_yticklabels([str(int(min(BETAS))), str(int(min(BETAS)/2)), '0', str(int(max(BETAS)/2)), str(int(max(BETAS)))])
ax4.set_yticks([0, 6, 10, 14, 20] )
ax4.set_yticklabels(['-25', '-10', '0', '10', '25'])
ax4.axvline(x=mu0_index, ls='dashed', color='black')
ax4.axhline(y=int(len(BETAS)/2), ls='dashed', color='black')
ax4.set_xlabel(r'$\mu$')
ax4.set_ylabel(r'$\beta$')
fig.colorbar(HM4, ax=ax4, label=r'Transfer entropy (FM) (bits)')
beta_slice_color1, beta_slice_color2, beta_slice_color3 = 'blue', 'dodgerblue', 'cyan'
mu_slice_color = 'magenta'
beta_slice_mu_index1, beta_slice_mu_index2, beta_slice_mu_index3 = 11, 15, 19
mu_slice_beta_index = 12
ax4.axvline(x = beta_slice_mu_index1, ls='dotted', color=beta_slice_color1)
ax4.axvline(x = beta_slice_mu_index2, ls='dotted', color=beta_slice_color2)
ax4.axvline(x = beta_slice_mu_index3, ls='dotted', color=beta_slice_color3)
ax4.axhline(y = mu_slice_beta_index, ls='dotted', color=mu_slice_color)
ax9.plot(BETAS, TE[:, beta_slice_mu_index1], "-", color=beta_slice_color1)
ax9.plot(BETAS, TE[:, beta_slice_mu_index2], "-", color=beta_slice_color2)
ax9.plot(BETAS, TE[:, beta_slice_mu_index3], "-", color=beta_slice_color3)
ax10.plot(MUS, TE[mu_slice_beta_index, :], "-", color=mu_slice_color)
ax9.set_xlabel(r'$\beta$')
ax9.set_ylabel(r'$TE_{FM}$ (bits)')
ax9.set_ylim(0.1, 0.37)
ax10.set_xlabel(r'$\mu$')
ax10.set_ylabel(r'$TE_{FM}$ (bits)')
ax10.set_xlim(-1, 2)
ax10.set_xticks([-1, 0, 1, 2])
ax10.set_ylim(0.1, 0.37)


BETAS = np.linspace(-25, 25, 21)

HM6 = ax6.imshow(TE_am, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto', vmin=0.1, vmax=0.25)  #vmin=0
ax6.set_xticks([0,   7,   14,  len(MUS)-1])
ax6.set_xticklabels([str(int(min(MUS))), '0', '1' ,str(int(max(MUS)))])
#ax6.set_yticks([0, int(1*len(BETAS)/4), int(len(BETAS)/2), int(3*len(BETAS)/4), len(BETAS)-1] )
#ax6.set_yticklabels([str(int(min(BETAS))), str(int(min(BETAS)/2)), '0', str(int(max(BETAS)/2)), str(int(max(BETAS)))])
ax6.set_yticks([0, 6, 10, 14, 20] )
ax6.set_yticklabels(['-25', '-10', '0', '10', '25'])
ax6.axvline(x=mu0_index, ls='dashed', color='black')
ax6.axhline(y=int(len(BETAS)/2), ls='dashed', color='black')
ax6.set_xlabel(r'$\mu$')
ax6.set_ylabel(r'$\beta$')
fig.colorbar(HM6, ax=ax6, label=r'Transfer entropy (AM) (bits)')
beta_slice_color1, beta_slice_color2, beta_slice_color3 = 'blue', 'dodgerblue', 'cyan'
mu_slice_color1, mu_slice_color2 = 'magenta', 'orange'
beta_slice_mu_index1, beta_slice_mu_index2, beta_slice_mu_index3 = 11, 15, 19
mu_slice_beta_index1, mu_slice_beta_index2 = 11, 15
ax6.axvline(x = beta_slice_mu_index1, ls='dotted', color=beta_slice_color1)
ax6.axvline(x = beta_slice_mu_index2, ls='dotted', color=beta_slice_color2)
ax6.axvline(x = beta_slice_mu_index3, ls='dotted', color=beta_slice_color3)
ax6.axhline(y = mu_slice_beta_index1, ls='dotted', color=mu_slice_color1)
ax6.axhline(y = mu_slice_beta_index2, ls='dotted', color=mu_slice_color2)
ax11.plot(BETAS, TE_am[:, beta_slice_mu_index1], "-", color=beta_slice_color1)
ax11.plot(BETAS, TE_am[:, beta_slice_mu_index2], "-", color=beta_slice_color2)
ax11.plot(BETAS, TE_am[:, beta_slice_mu_index3], "-", color=beta_slice_color3)
ax12.plot(MUS,   TE_am[mu_slice_beta_index1, :], "-", color=mu_slice_color1)
ax12.plot(MUS,   TE_am[mu_slice_beta_index2, :], "-", color=mu_slice_color2)
ax11.set_xlabel(r'$\beta$')
ax11.set_ylabel(r'$TE_{AM}$ (bits)')
ax11.set_ylim(0.05, 0.25)
ax12.set_xlabel(r'$\mu$')
ax12.set_ylabel(r'$TE_{AM}$ (bits)')
ax12.set_xlim(-1, 2)
ax12.set_xticks([-1, 0, 1, 2])
ax12.set_ylim(0.05, 0.25)


ax1.text(-0.15, 1.2, "A", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax2.text(-0.15, 1.2, "B", transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax3.text(-0.15, 1.2, "C", transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax4.text(-0.15, 1.2, "D", transform=ax4.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax5.text(-0.15, 1.2, "E", transform=ax5.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax6.text(-0.15, 1.2, "F", transform=ax6.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\Fig2.jpeg', dpi=300)





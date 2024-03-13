import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def Hopf_dots(mu, w0, alpha, beta, X, Y, Fx, Fy):
    X_dot = mu*X - w0*Y - (alpha*X - beta*Y)*(X**2 + Y**2) + Fx
    Y_dot = mu*Y + w0*X - (alpha*Y + beta*X)*(X**2 + Y**2) + Fy
    return X_dot, Y_dot

def Hopf_RK2(mu, w0, alpha, beta, X, Y, Fx, Fy, dt, X_noise, Y_noise):
    Xk1, Yk1 = Hopf_dots(mu, w0, alpha, beta, X                   , Y                   , Fx, Fy)
    Xk2, Yk2 = Hopf_dots(mu, w0, alpha, beta, X + Xk1*dt + X_noise, Y + Yk1*dt + Y_noise, Fx, Fy)
    new_X = X + (dt/2)*(Xk1 + Xk2) + X_noise
    new_Y = Y + (dt/2)*(Yk1 + Yk2) + Y_noise
    return new_X, new_Y

def HOPF(dt, mu, w0, alpha, beta, r0, phi0, X_noise_vector, Y_noise_vector):
    N = len(X_noise_vector)
    X = np.zeros(N, dtype=float)
    Y = np.zeros(N, dtype=float)
    X[0] = r0*np.cos(phi0)
    Y[0] = r0*np.sin(phi0)
    for i in range(1, N):
        X[i], Y[i] = Hopf_RK2(mu, w0, alpha, beta, X[i-1], Y[i-1], 0, 0, dt, X_noise_vector[i], Y_noise_vector[i])
    return X, Y

def analytic_Lyapunov(mu, beta, D):
    if mu <= 0:
        lyap = mu
    else:
        lyap = D*abs(beta)/mu
    return lyap

def redo_fit(tt, R_diff_avg, start_point, end_point):
    log_diff = np.log(R_diff_avg)
    fit_param, CoV = np.polyfit(tt[start_point:end_point], log_diff[start_point:end_point], 1, cov=True)  #m is in 1/cycles
    slope, intercept = fit_param
    lyap = slope/(2*np.pi)  #units of Omega0 if Omega0=1
    print(lyap)
    plt.figure()
    plt.plot(tt, log_diff, color='black')
    plt.plot(tt[start_point:end_point], slope*tt[start_point:end_point] + intercept, color='red')



def find_Lyapunov(mu, beta, num_cycles, num_IC, phi_diff):
    alpha = 1
    W0 = 1
    D = 0.1   #0.1
    #num_IC = 100  #300  20000
    dt = 2*np.pi*10**(-4)  #-4
    lyap_analytic = analytic_Lyapunov(mu, beta, D)
    #num_cycles = 0.5
    N = int(round(num_cycles*2*np.pi/dt))
    tt = np.linspace(0, num_cycles, N)  #time in units of cycle
    if mu > 0:
        #r_ic = (mu/alpha)**0.5  #or 0.01
        w0 = W0 + beta*mu/alpha
    else:
        #r_ic = 0.01
        w0 = W0
    R_diff = np.zeros((N, num_IC))
    x_test, y_test = HOPF(dt, mu, w0, alpha, beta, 0, 0,   ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, 100000), ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, 100000))
    r_ic = ((x_test**2 + y_test**2)**0.5)[int(len(x_test)/4):].mean()
    #phi_diff = 2*np.pi/100000      #/ r_ic   # 2*np.pi/1000 or 2*np.pi/10000   #divide by r_ic to get fixed arclength

    for j in range(num_IC):
        x_nv = ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, N)
        y_nv = ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, N)
        x1, y1 = HOPF(dt, mu, w0, alpha, beta, r_ic, 0,        x_nv, y_nv)
        x2, y2 = HOPF(dt, mu, w0, alpha, beta, r_ic, phi_diff, x_nv, y_nv)
        R_diff[:, j] = ((x2-x1)**2 + (y2-y1)**2)**0.5
    R_diff_avg = R_diff.mean(axis=1)
    log_diff = np.log(R_diff_avg)
    fit_param, CoV = np.polyfit(tt, log_diff, 1, cov=True)  #m is in 1/cycles
    slope, intercept = fit_param
    lyap = slope/(2*np.pi)  #units of Omega0 if Omega0=1
    plt.figure()
    plt.plot(tt, log_diff, color='black')
    plt.plot(tt, slope*tt + intercept, color='red')
    print(lyap, lyap_analytic)
    return lyap, lyap_analytic, tt, R_diff_avg



'''
#FINDING MU slice
mu1 = 2
mu2 = 2.66667
mu3 = 3
beta = 5
lyap1, lyap_analytic1, tt1, R_diff_avg1 = find_Lyapunov(mu1, beta)
lyap2, lyap_analytic2, tt2, R_diff_avg2 = find_Lyapunov(mu2, beta)
lyap3, lyap_analytic3, tt3, R_diff_avg3 = find_Lyapunov(mu3, beta)
#redo_fit(tt1, R_diff_avg3, 5500)
'''

'''
#FINDING BETA slice
mu = 1
Beta_Slice = np.linspace(-10, 10, 13)
Lyaps, Lyap_analytics = np.zeros(len(Beta_Slice)), np.zeros(len(Beta_Slice))
for i in range(len(Beta_Slice)):
    print(i)
    Lyaps[i], Lyap_analytics[i], tt1, R_diff_avg = find_Lyapunov(mu, Beta_Slice[i])
#redo_fit(tt1, R_diff_avg3, 5500)
plt.figure()
plt.plot(Beta_Slice, Lyap_analytics)
plt.plot(Beta_Slice, Lyaps, "o")
'''



'''
#FINING HEAT MAP
num_IC = 300  #300 is about 4 hours
num_cycles = 0.5  #0.5
D = 0.1   #0.1 or 0.01
Mus = np.linspace(-1, 2, 22)  #22 pts
Betas = np.linspace(-25, 25, 21)
dt = 2*np.pi*10**(-4)  #-4
phi_diff = 2*np.pi/1000
N = int(round(num_cycles*2*np.pi/dt))
tt = np.linspace(0, num_cycles, N)  #time in units of cycle
LYAP = np.zeros(( len(Betas), len(Mus) ))
alpha = 1
W0 = 1
for m in range(len(Mus)):
    print(m)
    for b in range(len(Betas)):
        mu =   Mus[m]
        beta = Betas[b]
        w0 = W0 + beta*mu/alpha
        LYAP[b, m] = mu    #lyap=mu for mu<0
        if mu >= 0:
            r_ic = 0.01
            if mu > 0:
                r_ic = (mu/alpha)**0.5
            R_diff = np.zeros((N, num_IC))
            for j in range(num_IC):
                x_nv = ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, N)
                y_nv = ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, N)
                x1, y1 = HOPF(dt, mu, w0, alpha, beta, r_ic, 0,        x_nv, y_nv)
                x2, y2 = HOPF(dt, mu, w0, alpha, beta, r_ic, phi_diff, x_nv, y_nv)
                R_diff[:, j] = ((x2-x1)**2 + (y2-y1)**2)**0.5
            R_diff_avg = R_diff.mean(axis=1)
            log_diff = np.log(R_diff_avg)
            slope, intercept = np.polyfit(tt, log_diff, 1)  #m is in 1/cycles
            Lam = slope/(2*np.pi)  #units of Omega0
            LYAP[b, m] = Lam
for i in range(len(Mus)):
    if Mus[i] >= 0:
        mu0_index = i
        break
plt.figure()
plt.imshow(LYAP, cmap='coolwarm', interpolation='nearest', origin='lower', aspect='auto', vmin=-1, vmax=2)  #vmin=0
plt.xticks([0, mu0_index,   len(Mus)-1],   [str(min(Mus)), '0' ,str(max(Mus))] )
plt.yticks([0, int(len(Betas)/2), len(Betas)-1], [str(min(Betas)), '0' ,str(max(Betas))] )
plt.axvline(x=mu0_index, ls='dashed', color='black')
plt.axhline(y=int(len(Betas)/2), ls='dashed', color='black')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\beta$')
plt.colorbar(label=r'Lyapunov exponent ($\Omega_0$)')
'''

#np.save(r'C:\Users\Justin\Desktop\fig3_LYAP_beta25_mu2_0p5cycle_300ic', LYAP)



'''
#show data
plt.figure(figsize=(5,5))
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.figure()
for i in range(num_IC):
    plt.plot(R_diff[:,i])
plt.plot(R_diff_avg, color='black')
plt.ylim(0, 1.2*max(R_diff_avg))
plt.figure()
plt.plot(tt, log_diff, color='black')
plt.plot(tt, slope*tt + intercept, color='red')
print("Lyap =", Lam, "Omega0")
'''






'''
#Individual Points     #0, 1.66667,   3.3333,   5.,   6.6667,   8.3333  10
mu, beta = 3, 5
lyap, lyap_analytic, tt, R_diff_avg = find_Lyapunov(mu, beta, 2.0, 10000, 2*np.pi/10000)   #1 cycle, 1000IC, 2pi/10000 phi_diff
#redo_fit(tt, R_diff_avg, 2000, 5000)
'''




#LOADING TRACE DATA
LYAP = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig1_LYAP_beta10_mu2_0p5cycle_300ic.npy')

LYAP = gaussian_filter(LYAP, 1.5)  #sigma

slice_color1='magenta'
slice_color2='dodgerblue'
Mus = np.linspace(-1, 3, 13)
Lyaps1 = np.array([-1.057, -0.69 , -0.356, -0.003,  0.24 ,  0.391,  0.412,  0.397, 0.3,  0.222,  0.2131,  0.1422,  0.1347])  #beta=5 redid last 3 with 10k IC, variable num_cycles
Lyaps2 = np.array([ 0.9905,  0.8656,  0.6495,  0.4994,  0.2660, 0.0906,    -0.0621,
                    0.0892,  0.2922,  0.4651, 0.6766,  0.8555,  0.9861])

fig = plt.figure(figsize=(10, 3))
plt.subplots_adjust(left=0.1, right=0.92, bottom=0.2, top=0.85, wspace=0.4, hspace=0.3)
ax1  = plt.subplot2grid((1, 3), (0,0), colspan=1, rowspan=1)
ax2  = plt.subplot2grid((1, 3), (0,1), colspan=1, rowspan=1)
ax3  = plt.subplot2grid((1, 3), (0,2), colspan=1, rowspan=1)

ax1.plot(np.linspace(-1.5, 0, 100), np.linspace(-1.5, 0, 100), color='black', linewidth=2)
ax1.plot(np.linspace(0.1, 3.5, 100), 5*0.1/np.linspace(0.1, 3.5, 100), color='black', linewidth=2)
ax1.plot(Mus, Lyaps1, "o", color=slice_color1)
#ax1.errorbar(Mus, Lyaps, yerr=Lyaps_dev, ls='none', color=slice_color)
ax1.set_xlim(-1.3, 3.3)
ax1.set_ylim(-1.2, 1.0)
ax1.axvline(x=0, ls='dashed', color='black')
ax1.axhline(y=0, ls='dashed', color='black')
ax1.set_xlabel(r'$\mu$')
ax1.set_ylabel(r'Lyapunov exponent ($\Omega_0$)')

ax2.plot(np.linspace(-10, 10, 100), 0.1*abs(np.linspace(-10, 10, 100)), color='black', linewidth=2)
ax2.plot(np.linspace(-10, 10, 13), Lyaps2, "o", color=slice_color2)
ax2.axvline(x=0, ls='dashed', color='black')
ax2.axhline(y=0, ls='dashed', color='black')
ax2.set_xlabel(r'$\beta$')
ax2.set_ylabel(r'Lyapunov exponent ($\Omega_0$)')

MUS, BETAS, mu0_index = np.linspace(-1, 2, 22), np.linspace(-10, 10, 21), 7
HM = ax3.imshow(LYAP, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto', vmin=-1, vmax=1)  #vmin=0   #interpolation=none bilinear
ax3.set_xticks([0,   7,  14,  21])
ax3.set_xticklabels(['-1', '0', '1' ,'2'])
ax3.set_yticks([0, int(1*len(BETAS)/4), int(len(BETAS)/2), int(3*len(BETAS)/4), len(BETAS)-1] )
ax3.set_yticklabels([str(int(min(BETAS))), str(int(min(BETAS)/2)), '0', str(int(max(BETAS)/2)), str(int(max(BETAS)))])
#ax3.set_xlim(5-0.5, len(MUS)-1+0.5)
ax3.axvline(x=mu0_index, ls='dashed', color='black')
ax3.axhline(y=int(len(BETAS)/2), ls='dashed', color='black')
ax3.axhline(y=15, ls='dotted', color=slice_color1)
ax3.axvline(x=14, ls='dotted', color=slice_color2)
ax3.set_xlabel(r'$\mu$')
ax3.set_ylabel(r'$\beta$')
fig.colorbar(HM, ax=ax3, label=r'Lyapunov exponent ($\Omega_0$)')

ax1.text(-0.15, 1.15, "A", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax2.text(-0.15, 1.15, "B",  transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax3.text(-0.15, 1.15, "C",  transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\Fig1.jpeg', dpi=300)





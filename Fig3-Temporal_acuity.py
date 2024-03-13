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



'''
#FINDING TRACES
N_cut, N_pre, N_stim, N_post = 3000, 1000, 1000, 3000
dt = 0.001*2*np.pi
num_IC = 64
F = 5
D = 0.001
Fx = np.concatenate((np.zeros(N_cut), np.zeros(N_pre), F*np.ones(N_stim), np.zeros(N_post)))
Initial_Phases = np.linspace(0, 2*np.pi*(num_IC - 1)/num_IC, num_IC)  # Initial phase of detector
W0 = 1
alpha = 1
mu1 = 0
beta1 = 0
w0_1 = W0 + beta1*mu1/alpha
r0_1 = (mu1/alpha)**0.5
mu2 = 1
beta2 = 5
w0_2 = W0 + beta2*mu2/alpha
r0_2 = (mu2/alpha)**0.5
random.seed(15)
r1_avg = np.zeros(len(Fx))
r2_avg = np.zeros(len(Fx))
for i in range(num_IC):
    x1, y1 = HOPF(dt, mu1, w0_1, alpha, beta1, D, Fx, np.zeros(len(Fx)), r0_1, Initial_Phases[i])
    x2, y2 = HOPF(dt, mu2, w0_2, alpha, beta2, D, Fx, np.zeros(len(Fx)), r0_2, Initial_Phases[i])
    r1_avg += ((x1**2 + y1**2)**0.5)/num_IC
    r2_avg += ((x2**2 + y2**2)**0.5)/num_IC
threshold1 = r1_avg[N_cut:N_cut+N_pre].mean() + r1_avg[N_cut:N_cut+N_pre].std()
threshold2 = r2_avg[N_cut:N_cut+N_pre].mean() + r2_avg[N_cut:N_cut+N_pre].std()
plt.plot(r1_avg[N_cut:], color='black')
plt.axhline(y=threshold1, ls='dotted', color='black')
plt.plot(r2_avg[N_cut:], color='red')
plt.axhline(y=threshold2, ls='dotted', color='red')
plt.plot(0.5*Fx[N_cut:]/F + 2, color='blue')
'''
#np.save(r'C:\Users\Justin\Desktop\Criticality\Figures\fig5_traces', [r1_avg, r2_avg])
r1_avg, r2_avg = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig3_traces.npy')
N_cut, N_pre, N_stim, N_post = 3000, 1000, 1000, 3000
#threshold1, threshold2 = r1_avg[N_cut:N_cut+N_pre].mean() + r1_avg[N_cut:N_cut+N_pre].std(),   r2_avg[N_cut:N_cut+N_pre].mean() + r2_avg[N_cut:N_cut+N_pre].std()



'''
#Finding Noise Sweep Data for TAU OFF
def find_tau(mu, beta, D, num_IC):
    N_cut, N_pre, N_stim, N_post = 3000, 1000, 1000, 10000
    if D < 10**(-3):
        N_post = 100000
    dt = 0.001*2*np.pi
    F = 5
    Fx = np.concatenate((np.zeros(N_cut), np.zeros(N_pre), F*np.ones(N_stim), np.zeros(N_post)))
    Initial_Phases = np.linspace(0, 2*np.pi*(num_IC - 1)/num_IC, num_IC)  # Initial phase of detector
    W0 = 1
    alpha = 1
    w0 = W0 + beta*mu/alpha
    r0 = 0.0
    if mu > 0:
        r0 = (mu/alpha)**0.5
    r_avg = np.zeros(len(Fx))
    for i in range(num_IC):
        x, y = HOPF(dt, mu, w0, alpha, beta, D, Fx, np.zeros(len(Fx)), r0, Initial_Phases[i])
        r_avg += ((x**2 + y**2)**0.5)/num_IC
    threshold = r_avg[N_cut:N_cut+N_pre].mean() + r_avg[N_cut:N_cut+N_pre].std()
    tau = (dt/(2*np.pi))*N_post #max tau can be in num natural periods
    start_index = N_cut + N_pre + N_stim
    for i in range(N_post):
        if r_avg[start_index + i] < threshold:
            tau = (dt/(2*np.pi))*i
            break
    return tau

num_IC = 100  #32 at least
num_trials = 5  #5     32 and 5 trials took about an hour
mu1, mu2 = 0, 1
beta1, beta2 = 0, 5
Ds = np.logspace(-5, 0, 13)
#Ds = np.logspace(-3, 0, 10)
Tau1 = np.zeros(len(Ds))
Tau2 = np.zeros(len(Ds))
for d in range(len(Ds)):
    print(d)
    D = Ds[d]
    for trial in range(num_trials):
        Tau1[d] += find_tau(mu1, beta1, D, num_IC)/num_trials
        Tau2[d] += find_tau(mu2, beta2, D, num_IC)/num_trials
plt.figure()
plt.plot(Ds, Tau1, "o-", color='black')
plt.plot(Ds, Tau2, "o-", color='red')
plt.xscale('log')
plt.yscale('log')
'''
Ds = np.logspace(-5, 0, 13)  #from noise sweeps
Taus1 = np.array([1.00000e+02, 1.00000e+02, 1.00000e+02, 6.63054e+01, 6.54680e+00,
       3.39860e+00, 1.98420e+00, 1.38720e+00, 7.36200e-01, 4.16600e-01,
       2.47600e-01, 1.46200e-01, 7.78000e-02])
Taus2 = np.array([0.5312, 0.4916, 0.4876, 0.4222, 0.3704, 0.344 , 0.3024, 0.2854,
       0.2494, 0.2334, 0.1954, 0.105 , 0.0428])


'''
#Finding TAU OFF HEATMAP
def find_tau(mu, beta):
    D = 0.001
    num_IC = 64
    N_cut, N_pre, N_stim, N_post = 3000, 1000, 1000, 3000
    dt = 0.001*2*np.pi
    F = 5
    Fx = np.concatenate((np.zeros(N_cut), np.zeros(N_pre), F*np.ones(N_stim), np.zeros(N_post)))
    Initial_Phases = np.linspace(0, 2*np.pi*(num_IC - 1)/num_IC, num_IC)  # Initial phase of detector
    W0 = 1
    alpha = 1
    w0 = W0
    r0 = 0.0
    if mu > 0:
        r0 = (mu/alpha)**0.5
        w0 = W0 + beta*mu/alpha
    r_avg = np.zeros(len(Fx))
    for i in range(num_IC):
        x, y = HOPF(dt, mu, w0, alpha, beta, D, Fx, np.zeros(len(Fx)), r0, Initial_Phases[i])
        r_avg += ((x**2 + y**2)**0.5)/num_IC
    threshold = r_avg[N_cut:N_cut+N_pre].mean() + r_avg[N_cut:N_cut+N_pre].std()
    tau = (dt/(2*np.pi))*N_post #max tau can be in num natural periods
    start_index = N_cut + N_pre + N_stim
    for i in range(N_post):
        if r_avg[start_index + i] < threshold:
            tau = (dt/(2*np.pi))*(i+1)
            break
    return tau
Mus = np.linspace(-1, 2, 22)  #21
Betas = np.linspace(-10, 10, 21)  #21
TAU_OFF = np.zeros(( len(Betas), len(Mus) ))
num_trials = 10
for m in range(len(Mus)):
    print(m)
    for b in range(len(Betas)):
        mu =   Mus[m]
        beta = Betas[b]
        for trial in range(num_trials):
            TAU_OFF[b, m] += find_tau(mu, beta)/num_trials
plt.figure()
plt.imshow(1/TAU_OFF, cmap='bwr', interpolation='nearest', origin='lower', aspect='auto', vmin=0, vmax=5)  #vmin=0
plt.xticks([0, int(len(Mus)/2),   len(Mus)-1],   [str(min(Mus)), '0' ,str(max(Mus))] )
plt.yticks([0, int(len(Betas)/2), len(Betas)-1], [str(min(Betas)), '0' ,str(max(Betas))] )
plt.axvline(x=int(len(Mus)/2), ls='dashed', color='black')
plt.axhline(y=int(len(Betas)/2), ls='dashed', color='black')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\beta$')
plt.colorbar(label='1/Tau')
'''
#np.save(r'C:\Users\Justin\Desktop\fig5_Tau_off_heatmap_beta10_mu2_64ic_10trials', TAU_OFF)






#Finding TAU ON
def find_tau_on(mu, beta, D, num_IC, plot='off'):
    N_pre, N_stim, N_stats = 1000, 5000, 5000   #1k 5k 5k  or  1k 2k 2k and avg over 10 trials
    dt = 0.001*2*np.pi
    F = 5  #1
    Fx = np.concatenate(( np.zeros(N_pre), F*np.ones(N_stim), F*np.ones(N_stats)))
    Initial_Phases = np.linspace(0, 2*np.pi*(num_IC - 1)/num_IC, num_IC)  # Initial phase of detector
    W0 = 1
    alpha = 1
    r0 = 0.0
    if mu > 0:
        r0 = (mu/alpha)**0.5
    w0 = W0 + beta*r0**2
    x_avg = np.zeros(len(Fx))
    for i in range(num_IC):
        x, y = HOPF(dt, mu, w0, alpha, beta, D, Fx, np.zeros(len(Fx)), r0, Initial_Phases[i])
        x_avg += x/num_IC
    x_stats = x_avg[-N_stats:]
    threshold_high = x_stats.mean() + 5*x_stats.std()
    threshold_low  = x_stats.mean() - 5*x_stats.std()
    tau_on = 1 * dt/(2*np.pi)  #smallest possible time
    for i in range(N_stim):
        if x_avg[N_pre + N_stim -i] > threshold_high or x_avg[N_pre + N_stim -i] < threshold_low:
            tau_on = (dt/(2*np.pi))*(N_stim - i)
            break
    #print(tau_on)
    if plot == 'on':
        plt.plot(x_avg)
        plt.axhline(y=threshold_high)
        plt.axhline(y=threshold_low)
    return tau_on


'''
#NOISE SWEEP for TAU ON.   USED F=1, N=2k, 2k
num_trials = 10
num_IC = 300
Ds = np.logspace(-5, 0, 13)   #13 points
#Ds = np.array([ 1.77827941e-04 ])
Taus_on1 = np.zeros(len(Ds))
Taus_on2 = np.zeros(len(Ds))
for trial in range(num_trials):
    for d in range(len(Ds)):
        print(trial, d)
        Taus_on1[d] += find_tau_on(0, 0, Ds[d], num_IC, plot='off')/num_trials
        #Taus_on2[d] += find_tau_on(1, 5, Ds[d], num_IC, plot='off')/num_trials
plt.figure()
plt.plot(Ds, Taus_on1, "o-", color='black')
plt.plot(Ds, Taus_on2, "o-", color='red')
plt.xscale("log")
#find_tau_on(mu, beta, D, num_IC, plot='on')
'''
Taus_ON1 = np.array([0.7868, 0.7673, 0.7534, 0.721, 0.6868, 0.6476, 0.6111, 0.5628, 0.5152, 0.4377, 0.3167, 0.0777, 0.0472])
Taus_ON2 = np.array([0.6464, 0.5643, 0.4812, 0.4229, 0.3727, 0.3612, 0.3287, 0.2889, 0.2562, 0.1311, 0.0245, 0.0014, 0.001 ])


'''
#Heatmap for TAU ON,   USE F = 5, N=5k, 5k
D = 0.001
num_trials = 2
num_IC = 300
Mus = np.linspace(-1, 2, 22)   #22
Betas = np.linspace(-10, 10, 21)  #21
TAU_ON = np.zeros(( len(Betas), len(Mus) ))
for m in range(len(Mus)):
    print(m)
    for b in range(len(Betas)):
        mu =   Mus[m]
        beta = Betas[b]
        TAU_ON[b, m] += find_tau_on(mu, beta, D, num_IC)/num_trials
plt.figure()
plt.imshow(1/TAU_ON, cmap='bwr', interpolation='nearest', origin='lower', aspect='auto')  #vmin=0
plt.xticks([0, int(len(Mus)/2),   len(Mus)-1],   [str(min(Mus)), '0' ,str(max(Mus))] )
plt.yticks([0, int(len(Betas)/2), len(Betas)-1], [str(min(Betas)), '0' ,str(max(Betas))] )
plt.axvline(x=int(len(Mus)/2), ls='dashed', color='black')
plt.axhline(y=int(len(Betas)/2), ls='dashed', color='black')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\beta$')
plt.colorbar(label='1/Tau_On')
plt.figure()
plt.imshow(TAU_ON, cmap='bwr', interpolation='nearest', origin='lower', aspect='auto')  #vmin=0
plt.xticks([0, int(len(Mus)/2),   len(Mus)-1],   [str(min(Mus)), '0' ,str(max(Mus))] )
plt.yticks([0, int(len(Betas)/2), len(Betas)-1], [str(min(Betas)), '0' ,str(max(Betas))] )
plt.axvline(x=int(len(Mus)/2), ls='dashed', color='black')
plt.axhline(y=int(len(Betas)/2), ls='dashed', color='black')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\beta$')
plt.colorbar(label='Tau_On')
#find_tau_on(mu, beta, D, num_IC, plot='on')
'''
#np.save(r'C:\Users\Justin\Desktop\fig5_Tau_on_heatmap', TAU_ON)



#T1 = np.load(r'C:\Users\Justin\Desktop\fig5_Tau_on_heatmap_300IC_2trials_1.npy')
#T2 = np.load(r'C:\Users\Justin\Desktop\fig5_Tau_on_heatmap_300IC_2trials_2.npy')
#T3 = np.load(r'C:\Users\Justin\Desktop\fig5_Tau_on_heatmap_300IC_2trials_3.npy')
#T_avg = (T1 + T2 + T3)/3
#np.save(r'C:\Users\Justin\Desktop\fig5_Tau_on_heatmap_300IC_6trials_avg', T_avg)



TAUS_OFF    = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig3_Tau_off_heatmap_beta10_mu2_64ic_10trials.npy')
TAUS_ON     = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig3_Tau_on_heatmap_300IC_6trials_avg.npy')
TAUS_OFF = gaussian_filter(TAUS_OFF, 2)
TAUS_ON = gaussian_filter(TAUS_ON, 3)

fig = plt.figure(figsize=(10, 5))
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.9, wspace=0.45, hspace=0.45)
ax1  = plt.subplot2grid((2, 3), (0,0), colspan=1, rowspan=1)
ax2  = plt.subplot2grid((2, 3), (0,1), colspan=1, rowspan=1)
ax3  = plt.subplot2grid((2, 3), (0,2), colspan=1, rowspan=1)
ax4  = plt.subplot2grid((2, 3), (1,0), colspan=1, rowspan=1)
ax5  = plt.subplot2grid((2, 3), (1,1), colspan=1, rowspan=1)
ax6  = plt.subplot2grid((2, 3), (1,2), colspan=1, rowspan=1)

tt1 = np.linspace(0, 5, len(r1_avg[N_cut:]))
ax1.plot(tt1, r1_avg[N_cut:], color='black')
ax1.axhline(y=r1_avg[N_cut:N_cut+N_pre].mean(), ls='dotted', color='black')
ax1.plot(tt1, r2_avg[N_cut:], color='red')
ax1.axhline(y=r2_avg[N_cut:N_cut+N_pre].mean(), ls='dotted', color='red')
ax1.plot(tt1, np.concatenate( (np.zeros(N_pre)+1.8, np.zeros(N_stim)+2, np.zeros(N_post)+1.8  ))   ,  "--",  color='blue')
ax1.set_xticks([0, 1, 2, 3, 4, 5])
ax1.set_ylim(0, 2.1)
ax1.set_xlabel("Time (cycles)")
ax1.set_ylabel(r'$r(t)$')

ax2.plot(Ds, Taus_ON1, "s--", color="black", fillstyle='none')
ax2.plot(Ds[:-1], Taus_ON2[:-1], "s--", color="red", fillstyle='none')
ax2.plot(Ds[3:], Taus1[3:], "o-", color="black")
ax2.plot(Ds, Taus2, "o-", color="red")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.axvspan(0.0002, 0.005, alpha=0.2, color='black')   #biological noise 1-5% of amplitude
ax2.set_xticks([10**(-5), 10**(-4), 10**(-3), 10**(-2), 0.1, 1])
ax2.set_yticks(      [0.001,  0.01,    0.1,   1, 10, 100])
ax2.set_yticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
ax2.set_ylim(0.001, 100)
ax2.set_xlabel("Noise strength")
ax2.set_ylabel(r'$\tau$ (cycles)')

ax3.plot(Ds, Taus_ON1, "s--", color="black", fillstyle='none')
ax3.plot(Ds[:-1], Taus_ON2[:-1], "s--", color="red", fillstyle='none')
ax3.plot(Ds[3:], Taus1[3:], "o-", color="black")
ax3.plot(Ds, Taus2, "o-", color="red")
ax3.set_xscale("log")
#ax3.set_yscale("log")
ax3.axvspan(0.0002, 0.005, alpha=0.2, color='black')   #biological noise 1-5% of amplitude
ax3.set_xticks([10**(-5), 10**(-4), 10**(-3), 10**(-2), 0.1, 1])
#ax3.set_yticks(      [0.001,  0.01,    0.1,   1, 10, 100])
#ax3.set_yticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
ax3.set_ylim(0, 1.5)
ax3.set_xlabel("Noise strength")
ax3.set_ylabel(r'$\tau$ (cycles)')

MUS, BETAS = np.linspace(-1, 2, 22), np.linspace(-10, 10, 21)
HM4 = ax4.imshow(1/TAUS_ON, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto', vmin=4, vmax=10)  #vmin=0
ax4.set_xticks([0,   7,   14,  len(MUS)-1])
ax4.set_xticklabels([str(int(min(MUS))), '0', '1' ,str(int(max(MUS)))])
#ax4.set_xticks([0,   5,   10,   15,  len(MUS)-1])
#ax4.set_xticklabels([str(int(min(MUS))), '-5', '0', '5' ,str(int(max(MUS)))])
ax4.set_yticks([0, int(1*len(BETAS)/4), int(len(BETAS)/2), int(3*len(BETAS)/4), len(BETAS)-1] )
ax4.set_yticklabels([str(int(min(BETAS))), str(int(min(BETAS)/2)), '0', str(int(max(BETAS)/2)), str(int(max(BETAS)))])
ax4.axvline(x=7, ls='dashed', color='black')
ax4.axhline(y=int(len(BETAS)/2), ls='dashed', color='black')
ax4.set_xlabel(r'$\mu$')
ax4.set_ylabel(r'$\beta$')
fig.colorbar(HM4, ax=ax4, label=r'$1/\tau_{on}$')

HM5 = ax5.imshow(1/TAUS_OFF, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=5)  #vmin=0
ax5.set_xticks([0,   7,   14,  len(MUS)-1])
ax5.set_xticklabels([str(int(min(MUS))), '0', '1' ,str(int(max(MUS)))])
ax5.set_yticks([0, int(1*len(BETAS)/4), int(len(BETAS)/2), int(3*len(BETAS)/4), len(BETAS)-1] )
ax5.set_yticklabels([str(int(min(BETAS))), str(int(min(BETAS)/2)), '0', str(int(max(BETAS)/2)), str(int(max(BETAS)))])
ax5.axvline(x=7, ls='dashed', color='black')
ax5.axhline(y=int(len(BETAS)/2), ls='dashed', color='black')
ax5.set_xlabel(r'$\mu$')
ax5.set_ylabel(r'$\beta$')
fig.colorbar(HM5, ax=ax5, label=r'$1/\tau_{off}$')

TAUS_MAX = np.zeros(( len(BETAS), len(MUS) ))
for b in range(len(BETAS)):
    for m in range(len(MUS)):
        #print(TAUS_ON[b, m]- TAUS_OFF[b, m])
        TAUS_MAX[b, m] = max(TAUS_ON[b, m], TAUS_OFF[b, m])
HM6 = ax6.imshow(1/TAUS_MAX, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=5)  #vmin=0
ax6.set_xticks([0,   7,   14,  len(MUS)-1])
ax6.set_xticklabels([str(int(min(MUS))), '0', '1' ,str(int(max(MUS)))])
ax6.set_yticks([0, int(1*len(BETAS)/4), int(len(BETAS)/2), int(3*len(BETAS)/4), len(BETAS)-1] )
ax6.set_yticklabels([str(int(min(BETAS))), str(int(min(BETAS)/2)), '0', str(int(max(BETAS)/2)), str(int(max(BETAS)))])
ax6.axvline(x=7, ls='dashed', color='black')
ax6.axhline(y=int(len(BETAS)/2), ls='dashed', color='black')
ax6.set_xlabel(r'$\mu$')
ax6.set_ylabel(r'$\beta$')
fig.colorbar(HM6, ax=ax6, label=r'$1/\tau_{max}$')

ax1.text(-0.16, 1.2, "A", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax2.text(-0.16, 1.2, "B", transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax3.text(-0.16, 1.2, "C", transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax4.text(-0.16, 1.2, "D", transform=ax4.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax5.text(-0.16, 1.2, "E", transform=ax5.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax6.text(-0.16, 1.2, "F", transform=ax6.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\Fig3.jpeg', dpi=300)


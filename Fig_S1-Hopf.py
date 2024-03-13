import random
import numpy as np
import matplotlib.pyplot as plt

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



#CHI FUNCTIONS

def sin_fit(x, dt, fit_w):  #fixed freq w: returns amp, phase and offset. fit_freq in units rad/unit time
    tList = np.linspace(0.0, 1.0*(len(x)-1)*dt, len(x))
    b = np.matrix(x, dtype='float').T
    rows = [ [np.sin(fit_w*t), np.cos(fit_w*t), 1] for t in tList]
    A = np.matrix(rows, dtype='float')
    (w,residuals,rank,sing_vals) = np.linalg.lstsq(A,b, rcond=0)
    amplitude = np.linalg.norm([w[0,0],w[1,0]],2)
    return amplitude

def Find_Chi(dt, mu, W0, alpha, beta, D, F=0.01, wf=1, num_IC=64, num_stim_cycles=20, num_cut_cycles=20):
    if mu > 0:
        r0 = (mu/alpha)**0.5
        w0 = W0 + beta*r0**2
    else:
        r0, w0 = 0, W0
    Initial_Phases = np.linspace(0, 2*np.pi*(num_IC - 1)/num_IC, num_IC)  # Initial phase of detector
    N = int(num_stim_cycles*2*np.pi/(wf*dt))
    N_cut = int(num_cut_cycles*2*np.pi/(wf*dt))
    tt = np.linspace(0, (N + N_cut - 1)*dt, N + N_cut)
    Fx = F*np.cos(wf*tt - np.pi/2)
    Fy = F*np.sin(wf*tt - np.pi/2)
    x_avg = np.zeros(N+N_cut)
    y_avg = np.zeros(N+N_cut)
    for ic in range(num_IC):
        x, y = HOPF(dt, mu, w0, alpha, beta, D, Fx, Fy, r0, Initial_Phases[ic])
        x_avg += x/num_IC
        y_avg += y/num_IC
    return (sin_fit(x_avg[N_cut:], dt, wf) + sin_fit(y_avg[N_cut:], dt, wf))/(2*F)

def Find_Gain(dt, mu, W0, alpha, beta, D, F=0.01, wf=1, num_IC=64, num_stim_cycles=20, num_cut_cycles=20):
    if mu > 0:
        r0 = (mu/alpha)**0.5
        w0 = W0 + beta*r0**2
    else:
        r0, w0 = 0, W0
    Initial_Phases = np.linspace(0, 2*np.pi*(num_IC - 1)/num_IC, num_IC)  # Initial phase of detector
    N = int(num_stim_cycles*2*np.pi/(wf*dt))
    N_cut = int(num_cut_cycles*2*np.pi/(wf*dt))
    tt = np.linspace(0, (N + N_cut - 1)*dt, N + N_cut)
    Fx = F*np.cos(wf*tt - np.pi/2)
    Fy = F*np.sin(wf*tt - np.pi/2)
    x_avg, y_avg, x0_avg, y0_avg = np.zeros(N + N_cut), np.zeros(N + N_cut), np.zeros(N + N_cut), np.zeros(N + N_cut)
    for ic in range(num_IC):
        x, y   = HOPF(dt, mu, w0, alpha, beta, D, Fx, Fy, r0, Initial_Phases[ic])
        x0, y0 = HOPF(dt, mu, w0, alpha, beta, D, np.zeros(len(Fx)), np.zeros(len(Fx)), r0, Initial_Phases[ic])
        x_avg  += x/num_IC
        y_avg  += y/num_IC
        x0_avg += x0/num_IC
        y0_avg += y0/num_IC
    r   = (x_avg[N_cut:]**2 + y_avg[N_cut:]**2)**0.5
    r0 = (x0_avg[N_cut:]**2 + y0_avg[N_cut:]**2)**0.5
    return r.mean()/r0.mean()

def find_bif_diagram(N, D, Mus):
    N_transient = 10000
    dt = 0.001*2*np.pi
    R0s = np.zeros(len(Mus))
    for i in range(len(Mus)):
        print(i)
        mu = Mus[i]
        if mu > 0:
            r0 = mu**0.5
        else:
            r0 = 0
        x, y = HOPF(dt, mu, 1, 1, 0, D, np.zeros(N), np.zeros(N), r0, 0)
        r = (x**2 + y**2)**0.5
        R0s[i] = r[N_transient:].mean()
    return np.round(R0s, 3)

'''
#Finding Bifurcation Smearing
N = 2000000
D = 0.1
Mus = np.linspace(-1, 1, 51)
Rs = find_bif_diagram(N, D, Mus)
plt.figure()
plt.plot(Mus, Rs, "-")
'''

Nc = 3000
dt = 0.001*2*np.pi
x, y = HOPF(dt, 1, 1, 1, 0, 0, np.zeros(Nc), np.zeros(Nc), 2.0, 0)  #mu=1 case
r_avg5 = (x**2 + y**2)**0.5 - 1   #DELTA R

'''
#Finding Critical Slowing Down
Dc1, Dc2, Dc3, Dc4 = 0.0, 0.001, 0.01, 0.1
num_IC1, num_IC2, num_IC3, num_IC4 = 1, 1000, 3000, 10000      #TIME COSTER RIGHT HERE.  Use 1, 1000, 3000, 10000
r_avg1 = np.zeros(Nc)
r_avg2 = np.zeros(Nc)
r_avg3 = np.zeros(Nc)
r_avg4 = np.zeros(Nc)
for i in range(num_IC1):
    x, y = HOPF(dt, 0, 1, 1, 0, Dc1, np.zeros(Nc), np.zeros(Nc), 1.0, 0)
    r_avg1 += ((x**2 + y**2)**0.5)/num_IC1
for i in range(num_IC2):
    x, y = HOPF(dt, 0, 1, 1, 0, Dc2, np.zeros(Nc), np.zeros(Nc), 1.0, 0)
    r_avg2 += ((x**2 + y**2)**0.5)/num_IC2
for i in range(num_IC3):
    x, y = HOPF(dt, 0, 1, 1, 0, Dc3, np.zeros(Nc), np.zeros(Nc), 1.0, 0)
    r_avg3 += ((x**2 + y**2)**0.5)/num_IC3
print("last one")
for i in range(num_IC4):
    x, y = HOPF(dt, 0, 1, 1, 0, Dc4, np.zeros(Nc), np.zeros(Nc), 1.0, 0)
    r_avg4 += ((x**2 + y**2)**0.5)/num_IC4
'''

#np.save(r'C:\Users\Justin\Desktop\Criticality\Figures_v1\fig1_traces', [r_avg1, r_avg2, r_avg3, r_avg4])
r_avg1, r_avg2, r_avg3, r_avg4 = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig_s1_traces.npy')


'''
#Finding t->inf amp levels for crit slowing down
Dc2, Dc3, Dc4 = 0.001, 0.01, 0.1
N_plat, N_plat_cut = 1100000, 100000
xD2, yD2 = HOPF(dt, 0, 1, 1, 0, Dc2, np.zeros(N_plat), np.zeros(N_plat), 0, 0)
xD3, yD3 = HOPF(dt, 0, 1, 1, 0, Dc3, np.zeros(N_plat), np.zeros(N_plat), 0, 0)
xD4, yD4 = HOPF(dt, 0, 1, 1, 0, Dc4, np.zeros(N_plat), np.zeros(N_plat), 0, 0)
r_plat2, r_plat3, r_plat4 = (xD2**2 + yD2**2)**0.5, (xD3**2 + yD3**2)**0.5, (xD4**2 + yD4**2)**0.5
R_Plats = np.array([ r_plat2[N_plat_cut:].mean(), r_plat3[N_plat_cut:].mean(), r_plat4[N_plat_cut:].mean() ])
print(R_Plats)
'''
deltaR_plat = np.array([0.17309411, 0.30923775, 0.54264447])  #D = 0.001, 0.01, 0.1



'''
#Finding Diverging GAIN Removal
num_trials = 100
num_IC = 1
F0 = 0.1
Ds = np.logspace(-8, 0, 20)
GAIN = np.zeros(len(Ds))
dt = 0.001*2*np.pi
mu, W0, alpha, beta = 0, 1, 1, 0
for trial in range(num_trials):
    for i in range(len(Ds)):
        print(trial, i)
        D = Ds[i]
        GAIN[i] += Find_Gain(dt, mu, W0, alpha, beta, D, F=F0, wf=1, num_IC=num_IC, num_stim_cycles=10, num_cut_cycles=10) / num_trials
plt.figure()
plt.plot(Ds, GAIN, "o-", color='black')
plt.xscale("log")
#plt.yscale("log")
'''



'''
#Finding Diverging CHI Removal
num_trials = 3
num_IC = 1024  #1024
Fs = np.logspace(-3, 1, 10)
Dc1, Dc2, Dc3, Dc4 = 0.0, 0.001, 0.01, 0.1
CHI1, CHI2, CHI3, CHI4 = np.zeros(len(Fs)), np.zeros(len(Fs)), np.zeros(len(Fs)), np.zeros(len(Fs))
dt = 0.001*2*np.pi
mu, W0, alpha, beta = 0, 1, 1, 0
for trial in range(num_trials):
    for i in range(len(Fs)):
        print(trial, i)
        F0 = Fs[i]
        CHI1[i] += Find_Chi(dt, mu, W0, alpha, beta, Dc1, F=F0, wf=1, num_IC=num_IC, num_stim_cycles=5, num_cut_cycles=5) / num_trials
        CHI2[i] += Find_Chi(dt, mu, W0, alpha, beta, Dc2, F=F0, wf=1, num_IC=num_IC, num_stim_cycles=5, num_cut_cycles=5) / num_trials
        CHI3[i] += Find_Chi(dt, mu, W0, alpha, beta, Dc3, F=F0, wf=1, num_IC=num_IC, num_stim_cycles=5, num_cut_cycles=5) / num_trials
        CHI4[i] += Find_Chi(dt, mu, W0, alpha, beta, Dc4, F=F0, wf=1, num_IC=num_IC, num_stim_cycles=5, num_cut_cycles=5) / num_trials
plt.figure()
plt.plot(Fs, CHI1, "o-", color='black')
plt.plot(Fs, CHI2, "o-", color='darkorange')
plt.plot(Fs, CHI3, "o-", color='red')
plt.plot(Fs, CHI4, "o-", color='darkviolet')
plt.xscale("log")
plt.yscale("log")
'''
'''
CHI1 = np.array([45.70867914, 39.22853611, 24.86286055, 12.91021451,  6.52844521,
        3.29999675,  1.6680812 ,  0.84318107,  0.42621082,  0.21544083])
CHI2 = np.array([17.91933395, 16.56824947, 15.60187877, 11.4076604 ,  6.32739338,
        3.27619371,  1.66483002,  0.84279619,  0.42617081,  0.21543522])
CHI3 = np.array([5.05641798, 5.87445488, 5.48985218, 5.37431002, 4.71606126,
       3.05394526, 1.63635105, 0.83945985, 0.42569676, 0.21537718])
CHI4 = np.array([6.698391  , 2.23343742, 1.99956103, 1.73561367, 1.79925293,
       1.70916255, 1.35802428, 0.80255741, 0.42106927, 0.2148048 ])
'''




#PANAL 1 - cartoon
dt = 0.001*2*np.pi
N = 10000
x1, y1 = HOPF(dt, -0.3, 1, 1, 0, 0, np.zeros(N), np.zeros(N), 1,    0.0)
x2, y2 = HOPF(dt, -0.1, 1, 1, 0, 0, np.zeros(N), np.zeros(N), 1,    0.0)
x3, y3 = HOPF(dt,    0, 1, 1, 0, 0, np.zeros(10*N),   np.zeros(10*N), 1,    0.0)
x4, y4 = HOPF(dt,    0.1, 1, 1, 0, 0, np.zeros(N),   np.zeros(N), 1,    0.0)
x4b, y4b = HOPF(dt,  0.1, 1, 1, 0, 0, np.zeros(N),   np.zeros(N), 0.1,    0.0)
x5, y5 = HOPF(dt,    0.3, 1, 1, 0, 0, np.zeros(N),   np.zeros(N), 1,    0.0)
x5b, y5b = HOPF(dt,  0.3, 1, 1, 0, 0, np.zeros(N),   np.zeros(N), 0.1,    0.0)



fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.3, hspace=0.2)
ax1  = plt.subplot2grid((5, 3), (0,0), colspan=3, rowspan=3)
ax2  = plt.subplot2grid((5, 3), (3,0), colspan=1, rowspan=2)
ax3  = plt.subplot2grid((5, 3), (3,1), colspan=1, rowspan=2)
ax4  = plt.subplot2grid((5, 3), (3,2), colspan=1, rowspan=2)
ax1.plot(x1-4, y1, "-", color='black', linewidth=0.5)
ax1.plot(x2-2, y2, "-", color='black', linewidth=0.5)
ax1.plot(x3, y3, "-", color='black', linewidth=0.5)
ax1.plot(x4+2, y4, "-", color='black', linewidth=0.5)
ax1.plot(x4b+2, y4b, "-", color='black', linewidth=0.5)
ax1.plot(x5+4, y5, "-", color='black', linewidth=0.5)
ax1.plot(x5b+4, y5b, "-", color='black', linewidth=0.5)
ax1.set_xlim(-5, 5.3)
ax1.set_ylim(-2.3, 1.5)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.arrow(1-4, 0, -0.1, 0.08, shape='full', lw=0.5, head_width=0.05, color='black')
ax1.arrow(1-2, 0, -0.1, 0.09, shape='full', lw=0.5, head_width=0.05, color='black')
ax1.arrow(1, 0, -0.1, 0.1, shape='full', lw=0.5, head_width=0.05, color='black')
ax1.arrow(1+2, 0, -0.1, 0.11, shape='full', lw=0.5, head_width=0.05, color='black')
ax1.arrow(1+4, 0, -0.1, 0.14, shape='full', lw=0.5, head_width=0.05, color='black')
pt1, pt2 = 250, 300
ax1.arrow(2+x4b[pt1], y4b[pt1], x4b[pt2]-x4b[pt1], y4b[pt2]-y4b[pt1],length_includes_head=True, shape='full', lw=0.5, head_width=0.05, color='black')
ax1.arrow(4+x5b[pt1], y5b[pt1], x5b[pt2]-x5b[pt1], y5b[pt2]-y5b[pt1],length_includes_head=True, shape='full', lw=0.5, head_width=0.05, color='black')
trace_size = 10000
scale_factor = 0.5
ax1.plot(np.linspace(0, 1, trace_size)-4.2, x1[:trace_size]*scale_factor - 1.5, color='black')
ax1.plot(np.linspace(0, 1, trace_size)-2.2, x2[:trace_size]*scale_factor - 1.5, color='black')
ax1.plot(np.linspace(0, 1, trace_size)-0.2, x3[:trace_size]*scale_factor - 1.5, color='black')
ax1.plot(np.linspace(0, 1, trace_size)+1.8, x4[:trace_size]*scale_factor - 1.5, color='black')
ax1.plot(np.linspace(0, 1, trace_size)+3.8, x5[:trace_size]*scale_factor - 1.5, color='black')
ax1.text(-3, 1, r'$\mu < 0$', color='black')
ax1.text(0, 1, r'$\mu = 0$', color='black')
ax1.text(3, 1, r'$\mu > 0$', color='black')
ax1.plot(np.array([-4.5, -4.5]), np.array([-2, -1]), color='black')
ax1.plot(np.array([-4.5, -3.5]), np.array([-2, -2]), color='black')
ax1.text(-4.3, -2.2, 'Time', color='black')
ax1.text(-4.9, -1.6, 'x(t)', color='black')
ax1.axis('off')




#PANAL 2
MUS = np.linspace(-1, 1, 51)
R1s = np.array([0.04 , 0.04 , 0.041, 0.041, 0.044, 0.044, 0.045, 0.047, 0.049,   #D=0p001, N=500k
       0.05 , 0.051, 0.053, 0.055, 0.057, 0.058, 0.06 , 0.067, 0.069,
       0.074, 0.082, 0.084, 0.093, 0.105, 0.118, 0.142, 0.17 , 0.221,
       0.271, 0.341, 0.394, 0.44 , 0.487, 0.527, 0.565, 0.6  , 0.631,
       0.664, 0.691, 0.721, 0.748, 0.774, 0.8  , 0.824, 0.847, 0.872,
       0.894, 0.917, 0.939, 0.959, 0.98 , 1.   ])
R2s = np.array([0.124, 0.125, 0.127, 0.132, 0.133, 0.136, 0.142, 0.143, 0.146,  #D=0p01  N=1M
       0.149, 0.156, 0.161, 0.166, 0.169, 0.178, 0.184, 0.191, 0.198,
       0.207, 0.215, 0.228, 0.238, 0.254, 0.268, 0.29 , 0.311, 0.328,
       0.361, 0.385, 0.421, 0.451, 0.487, 0.522, 0.555, 0.59 , 0.623,
       0.652, 0.684, 0.715, 0.744, 0.771, 0.792, 0.818, 0.844, 0.868,
       0.894, 0.913, 0.935, 0.956, 0.977, 0.998])
R3s = np.array([0.355, 0.358, 0.362, 0.367, 0.373, 0.379, 0.385, 0.391, 0.396,  #D=0.1 N=2M
       0.405, 0.411, 0.417, 0.426, 0.433, 0.443, 0.447, 0.458, 0.468,
       0.478, 0.487, 0.495, 0.506, 0.518, 0.526, 0.538, 0.551, 0.562,
       0.577, 0.59 , 0.6  , 0.618, 0.63 , 0.646, 0.664, 0.676, 0.695,
       0.71 , 0.728, 0.747, 0.767, 0.786, 0.806, 0.822, 0.846, 0.862,
       0.882, 0.905, 0.918, 0.945, 0.961, 0.98 ])
ax2.plot(np.linspace(-1, 0, 1000), np.zeros(1000), "--", color='black', zorder=5)
ax2.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000)**0.5, "--", color='black', zorder=5)
ax2.plot(MUS, R1s, color='orange')
ax2.plot(MUS, R2s, color='red')
ax2.plot(MUS, R3s, color='darkviolet')
ax2.set_xlim(-1, 1)
ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
ax2.set_ylim(-0.05, 1.0)
ax2.set_xlabel(r'$\mu$')
ax2.set_ylabel(r'$\langle r(t) \rangle$')


#PANAL 3
tt = np.linspace(0, (Nc-0)*dt/(2*np.pi) , Nc)   #num cycles
ax3.plot(tt, r_avg1, "--", color='black', zorder=5)
ax3.plot(tt, r_avg2, color='orange')
ax3.axhline(y=deltaR_plat[0], ls='dotted', color='orange')
ax3.plot(tt, r_avg3, color='red')
ax3.axhline(y=deltaR_plat[1], ls='dotted', color='red')
ax3.plot(tt, r_avg4, color='darkviolet')
ax3.axhline(y=deltaR_plat[2], ls='dotted', color='darkviolet')
ax3.plot(tt, r_avg5, "--", color='dodgerblue')
ax3.set_xlabel('Time (cycles)')
ax3.set_ylabel(r'$\Delta r(t)$')
ax3.set_xlim(0, 2)
ax3.set_ylim(0, 1.1)
#ax3.set_yscale("log")
ax3.set_xticks([0, 1, 2])

#PANEL 4
Ds = np.logspace(-8, 0, 20)
Gain = np.array([347.62625672, 193.16960068, 138.86929492,  81.96999612,
        49.146792  ,  30.14822997,  18.0881304 ,  12.56800235,
         8.56910618,   5.71962319,   4.39418064,   3.55364445,
         2.68383397,   2.06671152,   1.55080896,   1.22795141,
         1.07972843,   1.0211748 ,   1.00212627,   0.99535904])
ax4.axhline(y=1, ls='dotted', color='black')
ax4.plot(Ds, Gain, "o-", color='black')
ax4.set_xlabel('Noise strength')
ax4.set_ylabel('Gain')
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xticks([10**-8, 10**-6, 10**-4, 10**-2, 1])
ax4.set_xlim(0.3*10**-8, 3)
ax4.set_ylim(0.1, 1000)



ax1.text(-0.04, 1.0, "A", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax2.text(-0.1, 1.2, "B", transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax3.text(-0.05, 1.2, "C", transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax4.text(-0.05, 1.2, "D", transform=ax4.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
#plt.savefig(r'C:\Users\Justin\Desktop\FigS1.jpeg', dpi=300)





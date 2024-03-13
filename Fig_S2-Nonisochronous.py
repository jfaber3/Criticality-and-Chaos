import numpy as np
import matplotlib.pyplot as plt
import random

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


'''
#Finding Trace Data
np.random.seed(9)
num_cycles = 10
D = 0.1   #0.1 or 0.01
dt = 2*np.pi*10**(-3)  #-4 or -3
phase_perturbation = 2*np.pi/1000
N = int(round(num_cycles*2*np.pi/dt))
tt = np.linspace(0, num_cycles, N)  #time in units of cycle
alpha = 1
W0 = 1
mu = 1
beta = 10
w0 = W0 + beta*mu/alpha
x_nv = ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, N)
y_nv = ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, N)
x1, y1 = HOPF(dt, mu, w0, alpha, beta, (mu/alpha)**0.5, 0.0, x_nv, y_nv)
x2, y2 = HOPF(dt, mu, w0, alpha, beta, (mu/alpha)**0.5, phase_perturbation, x_nv, y_nv)
plt.figure()
plt.plot(tt, x1)
plt.plot(tt, x2)
'''



'''
#Finding Fractal Data
num_IC = 10000
num_cycles = 3
D = 0.1   #0.1 or 0.01
dt = 2*np.pi*10**(-3)  #-4 or -3
max_perturbation = 0.01
N = int(round(num_cycles*2*np.pi/dt))
tt = np.linspace(0, num_cycles, N)  #time in units of cycle
alpha = 1
W0 = 1
mu = 1
beta = 10
w0 = W0 + beta*mu/alpha
np.random.seed(24)  #24
x_nv = ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, N)
y_nv = ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, N)
X = np.zeros((num_IC, N))
Y = np.zeros((num_IC, N))
for j in range(num_IC):
    if j%100 == 0:
        print(j)
    radius, phase = max_perturbation*(np.random.random())**0.5, 2*np.pi*np.random.random()
    X[j, :], Y[j, :] = HOPF(dt, mu, w0, alpha, beta, (mu/alpha)**0.5 + radius*np.cos(phase), radius*np.sin(phase), x_nv, y_nv)
'''


'''
#Finding Log Divergence
np.random.seed(31)
num_IC = 1000
num_cycles = 5
D = 0.1   #0.1 or 0.01
dt = 2*np.pi*10**(-3)  #-4 or -3
phase_perturbation = 2*np.pi/1000
N = int(round(num_cycles*2*np.pi/dt))
tt = np.linspace(0, num_cycles, N)  #time in units of cycle
alpha = 1
W0 = 1
mu = 1
beta = 10
w0 = W0 + beta*mu/alpha
R_diff = np.zeros((num_IC, N))
for j in range(num_IC):
    x_nv = ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, N)
    y_nv = ((2*D*dt)**0.5)*np.random.normal(0.0, 1.0, N)
    x1, y1 = HOPF(dt, mu, w0, alpha, beta, (mu/alpha)**0.5, 0.0, x_nv, y_nv)
    x2, y2 = HOPF(dt, mu, w0, alpha, beta, (mu/alpha)**0.5, phase_perturbation, x_nv, y_nv)
    R_diff[j, :] = ((x2-x1)**2 + (y2-y1)**2)**0.5
R_diff_avg = R_diff.mean(axis=0)
log_diff = np.log(R_diff_avg)
slope, intercept = np.polyfit(tt, log_diff, 1)  #m is in 1/cycles
Lam = slope/(2*np.pi)  #units of Omega0
print(Lam)
plt.figure()
plt.plot(tt, R_diff_avg)
plt.figure()
plt.plot(tt, log_diff)
'''


#LOADING TRACE DATA
#np.save(r'C:\Users\Justin\Desktop\Criticality\Figures_v1\fig2_Trace_data', [x1, x2])
x_trace1, x_trace2 = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig_s2_Trace_data.npy')

#LOADING FRACTAL DATA
#np.save(r'C:\Users\Justin\Desktop\Criticality\Figures_v1\fig2_Fractal_data', [X, Y])
X, Y = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig_s2_Fractal_data.npy')
N = len(X[0,:])
num_cycles = int(N/1000)

#LOADING DIVERGENCE DATA
#np.save(r'C:\Users\Justin\Desktop\Criticality\Figures_v1\fig2_Divergence_data', [tt, R_diff])
tt, R_diff = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig_s2_Divergence_data.npy', allow_pickle=True)
fit_end = 500
log_diff = np.log(R_diff.mean(axis=0))
slope, intercept = np.polyfit(tt[:fit_end], log_diff[:fit_end], 1)





fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.07, right=0.95, bottom=0.1, top=0.95, wspace=0.3, hspace=0.5)
ax0  = plt.subplot2grid((5, 4), (0,0), colspan=4, rowspan=1)
ax1  = plt.subplot2grid((5, 4), (1,0), colspan=1, rowspan=2)
ax2  = plt.subplot2grid((5, 4), (1,1), colspan=1, rowspan=2)
ax3  = plt.subplot2grid((5, 4), (1,2), colspan=1, rowspan=2)
ax4  = plt.subplot2grid((5, 4), (1,3), colspan=1, rowspan=2)
ax5  = plt.subplot2grid((5, 4), (3,0), colspan=1, rowspan=2)
ax6  = plt.subplot2grid((5, 4), (3,1), colspan=1, rowspan=2)
ax7  = plt.subplot2grid((5, 4), (3,2), colspan=2, rowspan=2)

ax0.plot(np.linspace(0, 5, 5000), x_trace1[:5000], color='magenta')
ax0.plot(np.linspace(0, 5, 5000), x_trace2[:5000], color='black')
ax0.set_xlim(0, 5)
ax0.set_xlabel("Time (cycles)")
ax0.set_ylabel("x(t)")

t1, t2, t3, t4 = 0, 0.5, 0.8, 1
ax1.plot(X[:, int(N*t1/num_cycles)], Y[:, int(N*t1/num_cycles)], "o", color='black', markersize=1)
ax2.plot(X[:, int(N*t2/num_cycles)], Y[:, int(N*t2/num_cycles)], "o", color='black', markersize=1)
ax3.plot(X[:, int(N*t3/num_cycles)], Y[:, int(N*t3/num_cycles)], "o", color='black', markersize=1)
ax4.plot(X[:, int(N*t4/num_cycles)-1], Y[:, int(N*t4/num_cycles)-1], "o", color='black', markersize=1)
ax1.set_xlim(-1.5, 1.5)
ax2.set_xlim(-1.5, 1.5)
ax3.set_xlim(-1.5, 1.5)
ax4.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)
ax1.text(-1.4, -1.4, r'$t=$'+str(t1))
ax2.text(-1.4, -1.4, r'$t=$'+str(t2))
ax3.text(-1.4, -1.4, r'$t=$'+str(t3))
ax4.text(-1.4, -1.4, r'$t=$'+str(t4))
ax1.plot(np.cos(np.linspace(0, 2*np.pi, 1000)), np.sin(np.linspace(0, 2*np.pi, 1000)), color='blue', alpha=0.1)
ax2.plot(np.cos(np.linspace(0, 2*np.pi, 1000)), np.sin(np.linspace(0, 2*np.pi, 1000)), color='blue', alpha=0.1)
ax3.plot(np.cos(np.linspace(0, 2*np.pi, 1000)), np.sin(np.linspace(0, 2*np.pi, 1000)), color='blue', alpha=0.1)
ax4.plot(np.cos(np.linspace(0, 2*np.pi, 1000)), np.sin(np.linspace(0, 2*np.pi, 1000)), color='blue', alpha=0.1)
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')


ax5.axis('off')
phi = np.linspace(0, 2*np.pi, 1000)
ax5.set_xlim(-1.4, 1.4)
ax5.set_ylim(-1.4, 1.4)
ax5.plot(np.cos(phi), np.sin(phi), "-", color='black')
ax5.plot(0, 0, "o", color='black', fillstyle='none')
phi1 = np.linspace(np.pi/2, np.pi/2 + 1.3, 1000)
phi2 = np.linspace(np.pi/2, np.pi/2 + 0.7, 1000)
phi3 = np.linspace(np.pi/2, np.pi/2 + 0.3, 1000)
phi4 = np.linspace(np.pi/2, np.pi/2 + 0.1, 1000)
r1, r2, r3, r4 = 0.8, 0.9, 1.1, 1.2
x1, y1 = r1*np.cos(phi1), r1*np.sin(phi1)
x2, y2 = r2*np.cos(phi2), r2*np.sin(phi2)
x3, y3 = r3*np.cos(phi3), r3*np.sin(phi3)
x4, y4 = r4*np.cos(phi4), r4*np.sin(phi4)
clr = 'black'
ax5.plot(x1, y1, color=clr)
ax5.plot(x2, y2, color=clr)
ax5.plot(x3, y3, color=clr)
ax5.plot(x4, y4, color=clr)
ax5.arrow(x1[-1], y1[-1], (x1[-1]-x1[-2]), (y1[-1]-y1[-2]), head_width=0.1, head_length=0.1, color=clr)
ax5.arrow(x2[-1], y2[-1], (x2[-1]-x2[-2]), (y2[-1]-y2[-2]), head_width=0.1, head_length=0.1, color=clr)
ax5.arrow(x3[-1], y3[-1], (x3[-1]-x3[-2]), (y3[-1]-y3[-2]), head_width=0.1, head_length=0.1, color=clr)
ax5.arrow(x4[-1], y4[-1], (x4[-1]-x4[-2]), (y4[-1]-y4[-2]), head_width=0.1, head_length=0.1, color=clr)
x1 *= -1
x2 *= -1
x3 *= -1
x4 *= -1
y1 *= -1
y2 *= -1
y3 *= -1
y4 *= -1
ax5.plot(x1, y1, color=clr)
ax5.plot(x2, y2, color=clr)
ax5.plot(x3, y3, color=clr)
ax5.plot(x4, y4, color=clr)
ax5.arrow(x1[-1], y1[-1], (x1[-1]-x1[-2]), (y1[-1]-y1[-2]), head_width=0.1, head_length=0.1, color=clr)
ax5.arrow(x2[-1], y2[-1], (x2[-1]-x2[-2]), (y2[-1]-y2[-2]), head_width=0.1, head_length=0.1, color=clr)
ax5.arrow(x3[-1], y3[-1], (x3[-1]-x3[-2]), (y3[-1]-y3[-2]), head_width=0.1, head_length=0.1, color=clr)
ax5.arrow(x4[-1], y4[-1], (x4[-1]-x4[-2]), (y4[-1]-y4[-2]), head_width=0.1, head_length=0.1, color=clr)
ax5.text(-1.3, -1.2, r'$\beta > 0$')



ax6.axis('off')
phi = np.linspace(0, 2*np.pi, 1000)
ax6.set_xlim(-1.4, 1.4)
ax6.set_ylim(-1.4, 1.4)
ax6.plot(np.cos(phi), np.sin(phi), "-", color='black')
ax6.plot(0, 0, "o", color='black', fillstyle='none')
phi1 = np.linspace(np.pi/2, np.pi/2 + 0.1, 1000)
phi2 = np.linspace(np.pi/2, np.pi/2 + 0.3, 1000)
phi3 = np.linspace(np.pi/2, np.pi/2 + 0.5, 1000)
phi4 = np.linspace(np.pi/2, np.pi/2 + 0.7, 1000)
r1, r2, r3, r4 = 0.8, 0.9, 1.1, 1.2
x1, y1 = r1*np.cos(phi1), r1*np.sin(phi1)
x2, y2 = r2*np.cos(phi2), r2*np.sin(phi2)
x3, y3 = r3*np.cos(phi3), r3*np.sin(phi3)
x4, y4 = r4*np.cos(phi4), r4*np.sin(phi4)
clr = 'black'
ax6.plot(x1, y1, color=clr)
ax6.plot(x2, y2, color=clr)
ax6.plot(x3, y3, color=clr)
ax6.plot(x4, y4, color=clr)
ax6.arrow(x1[-1], y1[-1], (x1[-1]-x1[-2]), (y1[-1]-y1[-2]), head_width=0.1, head_length=0.1, color=clr)
ax6.arrow(x2[-1], y2[-1], (x2[-1]-x2[-2]), (y2[-1]-y2[-2]), head_width=0.1, head_length=0.1, color=clr)
ax6.arrow(x3[-1], y3[-1], (x3[-1]-x3[-2]), (y3[-1]-y3[-2]), head_width=0.1, head_length=0.1, color=clr)
ax6.arrow(x4[-1], y4[-1], (x4[-1]-x4[-2]), (y4[-1]-y4[-2]), head_width=0.1, head_length=0.1, color=clr)
x1 *= -1
x2 *= -1
x3 *= -1
x4 *= -1
y1 *= -1
y2 *= -1
y3 *= -1
y4 *= -1
ax6.plot(x1, y1, color=clr)
ax6.plot(x2, y2, color=clr)
ax6.plot(x3, y3, color=clr)
ax6.plot(x4, y4, color=clr)
ax6.arrow(x1[-1], y1[-1], (x1[-1]-x1[-2]), (y1[-1]-y1[-2]), head_width=0.1, head_length=0.1, color=clr)
ax6.arrow(x2[-1], y2[-1], (x2[-1]-x2[-2]), (y2[-1]-y2[-2]), head_width=0.1, head_length=0.1, color=clr)
ax6.arrow(x3[-1], y3[-1], (x3[-1]-x3[-2]), (y3[-1]-y3[-2]), head_width=0.1, head_length=0.1, color=clr)
ax6.arrow(x4[-1], y4[-1], (x4[-1]-x4[-2]), (y4[-1]-y4[-2]), head_width=0.1, head_length=0.1, color=clr)
ax6.text(-1.3, -1.2, r'$\beta < 0$')



ax7.plot(tt, log_diff, color='black')
ax7.plot(tt[:1000], intercept + slope*tt[:1000], "--", color='red')
ax7.set_xlim(0, 5)
ax7.set_ylim(-6, 1)
ax7.set_xlabel("Time (cycles)")
ax7.set_ylabel(r'$\log(|\Delta z(t)|)$')


ax0.text(-0.01, 1.3, "A", transform=ax0.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax1.text(-0.04, 1.0, "B", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax5.text(-0.04, 1.0, "C", transform=ax5.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax7.text(-0.05, 1.1, "D", transform=ax7.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
#plt.savefig(r'C:\Users\Justin\Desktop\FigS2.jpeg', dpi=300)




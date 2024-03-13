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

def sin_fit(x, dt, fit_w):  #fixed freq w: returns amp, phase and offset. fit_freq in units rad/unit time
    tList = np.linspace(0.0, 1.0*(len(x)-1)*dt, len(x))
    b = np.matrix(x, dtype='float').T
    rows = [ [np.sin(fit_w*t), np.cos(fit_w*t), 1] for t in tList]
    A = np.matrix(rows, dtype='float')
    (w,residuals,rank,sing_vals) = np.linalg.lstsq(A,b, rcond=0)
    amplitude = np.linalg.norm([w[0,0],w[1,0]],2)
    return amplitude

def Find_Res_Amp(dt, mu, w0, alpha, beta, D, r0, F, wf, num_IC, num_stim_cycles, num_cut_cycles):
    Initial_Phases = np.linspace(0, 2*np.pi*(num_IC - 1)/num_IC, num_IC)  # Initial phase of detector
    N = int(num_stim_cycles*2*np.pi/(wf*dt))
    N_cut = int(num_cut_cycles*2*np.pi/(wf*dt))
    tt = np.linspace(0, (N + N_cut - 1)*dt, N + N_cut)
    Fx = F*np.cos(wf*tt - np.pi/2)
    Fy = F*np.sin(wf*tt - np.pi/2)
    x_avg, y_avg = np.zeros(N+N_cut), np.zeros(N+N_cut)
    for ic in range(num_IC):
        x, y = HOPF(dt, mu, w0, alpha, beta, D, Fx, Fy, r0, Initial_Phases[ic])
        x_avg += x/num_IC
        y_avg += y/num_IC
    return ( sin_fit(x_avg[N_cut:], dt, wf) + sin_fit(y_avg[N_cut:], dt, wf) ) / 2

def Find_dynamic_range_analytic(mu, delta_w, alpha, beta, inc_factor, R_small=1):
    if mu > 0:
        return 3 + np.emath.logn(inc_factor, (mu/abs(delta_w))*(1 + (beta/alpha)**2)  )
    else:
        return 3 + np.emath.logn(inc_factor, (R_small**2)*(  (alpha**2 + beta**2)/(mu**2 + delta_w**2)  )**0.5   )

def Find_expected_range(mu, delta_w, alpha, beta, inc_factor, R_small=1):
    if mu > 0:
        r0 = (mu/alpha)**0.5
        F_min_exp = (abs(delta_w)*r0) / ((1 + (beta/alpha)**2)**0.5)
        F_max_exp = (inc_factor*r0)**3 * alpha * (1 + (beta/alpha)**2)**0.5
    else:
        R_big = inc_factor*R_small
        F_min_exp = R_small*(mu**2 + delta_w**2)**0.5
        F_max_exp = (R_big**3)*(alpha**2 + beta**2)**0.5
    return F_min_exp, F_max_exp




'''
#FINDING RESPONSE CURVES
dt = 0.001*2*np.pi
mu = 1
alpha = 1
beta1 = 0
beta2 = 2
beta3 = 5
W0 = 1
r0 = (mu/alpha)**0.5
D = 0.0
wf = 1.0
num_IC = 1  #32,  16
num_stim_cycles = 10  #10
num_cut_cycles = 10   #10
Fs = np.logspace(-3, 2, 50)  #-3, 2, 50
R1, R2, R3 = np.zeros(len(Fs)), np.zeros(len(Fs)), np.zeros(len(Fs))
for i in range(len(Fs)):
    print(i)
    R1[i] = Find_Res_Amp(dt, mu, W0 + beta1*mu/alpha, alpha, beta1, D, r0, Fs[i], wf, num_IC, num_stim_cycles, num_cut_cycles)
    R2[i] = Find_Res_Amp(dt, mu, W0 + beta2*mu/alpha, alpha, beta2, D, r0, Fs[i], wf, num_IC, num_stim_cycles, num_cut_cycles)
    R3[i] = Find_Res_Amp(dt, mu, W0 + beta3*mu/alpha, alpha, beta3, D, r0, Fs[i], wf, num_IC, num_stim_cycles, num_cut_cycles)
R1, R2, R3 = np.round(R1, 4), np.round(R2, 4), np.round(R3, 4)
plt.figure()
plt.plot(Fs, R1, "-", color='blue')
plt.plot(Fs, R2, "-", color='darkorange')
plt.plot(Fs, R3, "-", color='red')
plt.plot(np.logspace(1, 2, 100), 1.5*(np.logspace(1, 2, 100)**(1/3)), color='magenta')
plt.plot(np.logspace(-3, -2.3, 100), 25*(np.logspace(-3, -2.3, 100)**(1)), color='black')
plt.xscale("log")
plt.yscale("log")
'''




Fs = np.logspace(-3, 2, 50)

#wf = 1, 16IC, 10cycle, 10cut
R1a = np.array([0.0473, 0.0598, 0.0756, 0.0956, 0.1206, 0.1521, 0.1914, 0.2403,
       0.3002, 0.3725, 0.4575, 0.5533, 0.6542, 0.748 , 0.8207, 0.8662,
       0.8922, 0.9256, 0.9767, 1.0212, 1.0487, 1.0632, 1.0784, 1.0967,
       1.1186, 1.1448, 1.1758, 1.2122, 1.2545, 1.3032, 1.3591, 1.4226,
       1.4942, 1.5747, 1.6645, 1.7643, 1.8748, 1.9967, 2.1307, 2.2778,
       2.4389, 2.6148, 2.8068, 3.016 , 3.2438, 3.4914, 3.7604, 4.0525,
       4.3695, 4.7133])
R2a = np.array([0.105 , 0.1325, 0.167 , 0.2099, 0.263 , 0.3276, 0.4047, 0.4937,
       0.5919, 0.6922, 0.7857, 0.8677, 0.9347, 0.9783, 0.9997, 1.0067,
       1.0094, 1.0119, 1.015 , 1.0189, 1.0237, 1.0297, 1.0372, 1.0464,
       1.0578, 1.0716, 1.0885, 1.1088, 1.1331, 1.162 , 1.196 , 1.2357,
       1.2817, 1.3345, 1.3947, 1.4628, 1.5394, 1.6252, 1.7207, 1.8266,
       1.9435, 2.0724, 2.2138, 2.3689, 2.5384, 2.7236, 2.9254, 3.1453,
       3.3844, 3.6444])
R3a = np.array([0.2346, 0.2931, 0.3637, 0.4468, 0.541 , 0.6433, 0.7496, 0.8508,
       0.9299, 0.9765, 0.9956, 1.0006, 1.0016, 1.0021, 1.0026, 1.0033,
       1.0042, 1.0053, 1.0067, 1.0084, 1.0106, 1.0133, 1.0168, 1.0211,
       1.0265, 1.0332, 1.0415, 1.0517, 1.0642, 1.0795, 1.098 , 1.1203,
       1.1468, 1.1781, 1.2148, 1.2575, 1.3068, 1.3631, 1.4272, 1.4994,
       1.5805, 1.671 , 1.7715, 1.8828, 2.0055, 2.1405, 2.2886, 2.4507,
       2.6279, 2.8212])

#wf = 1.01, 32IC, 10cycle, 10cut
R1b = np.array([0.0451, 0.057 , 0.072 , 0.091 , 0.1149, 0.145 , 0.1826, 0.2295,
       0.2872, 0.3574, 0.4408, 0.5363, 0.6401, 0.7446, 0.8419, 0.9257,
       0.9639, 1.0023, 1.0303, 1.0405, 1.0507, 1.0632, 1.0783, 1.0966,
       1.1185, 1.1448, 1.1758, 1.2121, 1.2544, 1.3032, 1.3591, 1.4226,
       1.4942, 1.5747, 1.6645, 1.7643, 1.8748, 1.9967, 2.1307, 2.2778,
       2.4388, 2.6148, 2.8068, 3.016 , 3.2438, 3.4914, 3.7604, 4.0525,
       4.3695, 4.7133])
R2b = np.array([0.0987, 0.1246, 0.1571, 0.1977, 0.2481, 0.31  , 0.3845, 0.4718,
       0.57  , 0.6736, 0.7742, 0.8546, 0.9325, 0.9795, 0.9953, 1.0001,
       1.0001, 1.0099, 1.013 , 1.0169, 1.0218, 1.0278, 1.0353, 1.0446,
       1.056 , 1.0699, 1.0868, 1.1072, 1.1316, 1.1605, 1.1946, 1.2344,
       1.2804, 1.3333, 1.3935, 1.4617, 1.5384, 1.6242, 1.7198, 1.8258,
       1.9428, 2.0717, 2.2132, 2.3683, 2.5379, 2.7231, 2.925 , 3.1448,
       3.384 , 3.644 ])
R3b = np.array([0.2214, 0.2772, 0.3452, 0.426 , 0.5189, 0.6203, 0.7228, 0.8168,
       0.9013, 0.9461, 0.9384, 0.9963, 1.0005, 1.0011, 1.0016, 1.0023,
       1.0032, 1.0043, 1.0057, 1.0075, 1.0097, 1.0124, 1.0159, 1.0202,
       1.0256, 1.0323, 1.0406, 1.0508, 1.0634, 1.0787, 1.0972, 1.1195,
       1.146 , 1.1774, 1.2141, 1.2569, 1.3062, 1.3626, 1.4266, 1.4989,
       1.58  , 1.6705, 1.7711, 1.8824, 2.0052, 2.1402, 2.2883, 2.4505,
       2.6277, 2.821 ])

#wf = 1.1, 64IC, 10cycle, 10cut
R1c = np.array([0.0054, 0.0068, 0.0086, 0.0108, 0.0137, 0.0174, 0.022 , 0.0278,
       0.0352, 0.0445, 0.0564, 0.0714, 0.0906, 0.1149, 0.1457, 0.1837,
       0.2264, 0.2603, 0.2719, 0.7457, 1.0034, 1.0431, 1.0637, 1.0857,
       1.1102, 1.1384, 1.1709, 1.2083, 1.2515, 1.301 , 1.3573, 1.4212,
       1.4932, 1.5738, 1.6638, 1.7638, 1.8744, 1.9964, 2.1305, 2.2776,
       2.4387, 2.6147, 2.8067, 3.016 , 3.2437, 3.4913, 3.7604, 4.0525,
       4.3695, 4.7133])
R2c = np.array([0.0114, 0.0145, 0.0183, 0.0232, 0.0293, 0.0371, 0.047 , 0.0595,
       0.0753, 0.0955, 0.121 , 0.1528, 0.1901, 0.2264, 0.2425, 0.3376,
       0.9421, 0.9866, 0.9918, 0.9967, 1.0024, 1.0091, 1.0172, 1.027 ,
       1.0389, 1.0534, 1.0709, 1.0919, 1.1169, 1.1465, 1.1813, 1.2218,
       1.2685, 1.3221, 1.383 , 1.4519, 1.5293, 1.6158, 1.7119, 1.8184,
       1.936 , 2.0654, 2.2074, 2.3629, 2.5329, 2.7185, 2.9207, 3.1409,
       3.3804, 3.6407])
R3c = np.array([0.0267, 0.0337, 0.0427, 0.0541, 0.0685, 0.0868, 0.11  , 0.1391,
       0.1746, 0.2134, 0.2425, 0.2546, 0.7179, 0.9767, 0.9921, 0.9931,
       0.9941, 0.9954, 0.9968, 0.9987, 1.001 , 1.0038, 1.0073, 1.0118,
       1.0173, 1.0241, 1.0325, 1.0429, 1.0557, 1.0712, 1.0899, 1.1124,
       1.1392, 1.1709, 1.208 , 1.2511, 1.3007, 1.3574, 1.4218, 1.4944,
       1.5758, 1.6666, 1.7675, 1.879 , 2.0021, 2.1373, 2.2857, 2.448 ,
       2.6254, 2.819 ])


fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.9, wspace=0.5, hspace=0.35)
ax1  = plt.subplot2grid((2, 3), (0,0), colspan=1, rowspan=1)
ax2  = plt.subplot2grid((2, 3), (0,1), colspan=1, rowspan=1)
ax3  = plt.subplot2grid((2, 3), (0,2), colspan=1, rowspan=1)
ax4  = plt.subplot2grid((2, 3), (1,0), colspan=1, rowspan=1)
ax5  = plt.subplot2grid((2, 3), (1,1), colspan=1, rowspan=1)
ax6  = plt.subplot2grid((2, 3), (1,2), colspan=1, rowspan=1)


ax1.plot(Fs, R1a, "-", color='blue')
ax1.plot(Fs, R2a, "-", color='darkorange')
ax1.plot(Fs, R3a, "-", color='red')
ax1.plot(np.logspace(0.8, 2, 100), 1.5*(np.logspace(0.8, 2, 100)**(1/3)), "--", color='magenta')
ax1.plot(np.logspace(-3, -2, 100), 25*(np.logspace(-3, -2, 100)**(1)), "--", color='black')
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks([0.001, 0.01, 0.1, 1, 10, 100])
ax1.set_ylim(0.001, 10)
ax1.set_xlabel(r'$F$')
ax1.set_ylabel("Response amplitude")

ax2.plot(Fs, R1b, "-", color='blue')
ax2.plot(Fs, R2b, "-", color='darkorange')
ax2.plot(Fs, R3b, "-", color='red')
ax2.plot(np.logspace(0.8, 2, 100), 1.5*(np.logspace(0.8, 2, 100)**(1/3)), "--", color='magenta')
ax2.plot(np.logspace(-3, -2, 100), 25*(np.logspace(-3, -2, 100)**(1)), "--", color='black')
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xticks([0.001, 0.01, 0.1, 1, 10, 100])
ax2.set_ylim(0.001, 10)
ax2.set_xlabel(r'$F$')
ax2.set_ylabel("Response amplitude")

ax3.plot(Fs, R1c, "-", color='blue')
ax3.plot(Fs, R2c, "-", color='darkorange')
ax3.plot(Fs, R3c, "-", color='red')
ax3.plot(np.logspace(0.8, 2, 100), 1.5*(np.logspace(0.8, 2, 100)**(1/3)), "--", color='magenta')
ax3.plot(np.logspace(-3, -2, 100), 2*(np.logspace(-3, -2, 100)**(1)), "--", color='black')
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xticks([0.001, 0.01, 0.1, 1, 10, 100])
ax3.set_ylim(0.001, 10)
ax3.set_xlabel(r'$F$')
ax3.set_ylabel("Response amplitude")


R_ex = np.array([0.0451, 0.057 , 0.072 , 0.091 , 0.1149, 0.145 , 0.1826, 0.2295,
       0.2872, 0.3574, 0.4408, 0.5363, 0.6401, 0.7446, 0.8419, 0.9257,
       0.9639, 1.0023, 1.0303, 1.0405, 1.0507, 1.0632, 1.0783, 1.0966,
       1.1185, 1.1448, 1.1758, 1.2121, 1.2544, 1.3032, 1.3591, 1.4226,
       1.4942, 1.5747, 1.6645, 1.7643, 1.8748, 1.9967, 2.1307, 2.2778,
       2.4388, 2.6148, 2.8068, 3.016 , 3.2438, 3.4914, 3.7604, 4.0525,
       4.3695, 4.7133,    5.6826,  6.8617,  8.2941, 10.0298])  #R1b but added 4 more to extend range for illustration
index1, index2 = 13, 51
FF = np.concatenate((Fs, np.array([177.827941  ,  316.22776602,  562.34132519, 1000 ])))
ax4.plot(FF, R_ex, "-", color='black')
ax4.plot(FF[index1:index2+1], R_ex[index1:index2+1], "-", color='red', linewidth=3)
#ax4.plot([FF[index1], FF[index2]], [R_ex[index1], R_ex[index2]], "o", color='darkviolet')
#print(R_ex[index2]/R_ex[index1])
ax4.plot( [FF[index1], FF[index2]], [R_ex[index2], R_ex[index2]], ":", color='black')
ax4.plot( [FF[index1], FF[index2]], [R_ex[index1], R_ex[index1]], ":", color='black')
ax4.plot( [FF[index1], FF[index1]], [R_ex[index1], R_ex[index2]], ":", color='black')
ax4.plot( [FF[index2], FF[index2]], [R_ex[index1], R_ex[index2]], ":", color='black')
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
ax4.set_ylim(0.01, 20)
ax4.set_xlabel(r'$F$')
ax4.set_ylabel("Response amplitude")
ax4.text(0.22, 0.75, r'$10\times$', transform=ax4.transAxes, fontsize=10, fontweight='bold', va='top', ha='right', color='black')
ax4.text(0.63, 0.52, r'$10^{\gamma} \times$', transform=ax4.transAxes, fontsize=10, va='top', ha='right', color='black')


clr1, clr2, clr3 = 'red', 'green', 'blue'

Dyn_Range1 = np.array([6.98471174, 6.7936841 , 6.54941508, 6.2121537 , 5.68083169,  #D=0, delta_w = 0.01  #5 cycles
       4.96238419, 5.56792194, 6.06557307, 6.38879941, 6.62483185, 6.81010386])
Dyn_Range2 = np.array([2.62460415, 2.59296934, 2.55259746, 2.49697615, 2.40960414,
       2.29479872, 2.40913274, 2.49598593, 2.5510747 , 2.59090503, 2.62199156])
Dyn_Range3 = np.array([2.00069829, 1.96880112, 1.9280255 , 1.87176052, 1.78329899,
       1.66744117, 1.78330376, 1.87176033, 1.92802169, 1.9687942, 2.0006885 ])

Dyn_Range1_D4 = np.array([6.59703015, 6.76043162, 6.41265703, 6.21142976, 5.68871932,  #D=0.0001, delta_w = 0.01
       4.95765262, 5.55077363, 6.1122294 , 6.05870886, 6.63260705, 6.54278421])
Dyn_Range2_D4 = np.array([3.14056488, 2.86887606, 3.23119096, 3.56530847, 3.73367674,
       3.27541912, 3.46882492, 2.79276351, 3.13269473, 2.87374225, 3.02376823])
Dyn_Range3_D4 = np.array([2.02209907, 1.94581944, 1.93393788, 1.90234988, 1.81338889,
       1.67409038, 1.73298375, 1.84928938, 1.86950933, 2.00905954, 2.00768663])

Dyn_Range1_D3 = np.array([5.89757398, 5.98624672, 6.0524485 , 5.87769616, 5.26378912,  #D=0.001, delta_w = 0.01  #5 cycles
       5.23925188, 5.82358388, 6.07876057, 5.95092449, 5.7576917 ])
Dyn_Range2_D3 = np.array([3.20001206, 3.28299749, 3.37183778, 3.54837797, 3.75395635,
       3.76691098, 3.47637044, 3.2786726 , 3.26865287, 3.23689639])
Dyn_Range3_D3 = np.array([2.06522183, 2.04758537, 1.97575826, 1.93225914, 1.80418896,
       1.79274408, 1.91193581, 1.97681779, 2.04198009, 2.07710153])

DR = np.load(r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig5_Dyn_Rng_dw=0p01_D=0p001_1IC_20Trial_5cyc_5cut.npy')
DR = gaussian_filter(DR, 1)
Mus, Betas   = np.linspace( -1,  2, 22),  np.linspace(-10, 10, 21)

exp_Betas = np.linspace(-10, 10, 11)
smooth_Betas = np.linspace(-10, 10, 1000)
mu1, mu2, mu3 = 1, 0, -1
inc_factor1, inc_factor2, inc_factor3 = 10, 1000, 1000
delta_w = 0.01
ax5.plot(smooth_Betas, Find_dynamic_range_analytic(mu1, delta_w, 1, smooth_Betas, inc_factor1, R_small=0.01), color=clr1,  alpha=0.3)
ax5.plot(smooth_Betas, Find_dynamic_range_analytic(mu2, delta_w, 1, smooth_Betas, inc_factor2, R_small=0.01), color=clr2,  alpha=0.3)
ax5.plot(smooth_Betas, Find_dynamic_range_analytic(mu3, delta_w, 1, smooth_Betas, inc_factor3, R_small=0.01), color=clr3,  alpha=0.3)
ax5.plot(exp_Betas, Dyn_Range1, "o", color=clr1)
ax5.plot(exp_Betas, Dyn_Range2, "o", color=clr2)
ax5.plot(exp_Betas, Dyn_Range3, "o", color=clr3)
ax5.plot(np.linspace(-9, 9, 10), Dyn_Range1_D3, "+", color=clr1, fillstyle='none')
ax5.plot(np.linspace(-9, 9, 10), Dyn_Range2_D3, "+", color=clr2, fillstyle='none')
ax5.plot(np.linspace(-9, 9, 10), Dyn_Range3_D3, "+", color=clr3, fillstyle='none')
ax5.set_ylim(0, 8)
ax5.set_xlabel(r'$\beta$')
ax5.set_ylabel(r'Dynamic range ($\gamma$)')
ax5.text(0.9, 0.97, r'$\mu = 1$', transform=ax5.transAxes, fontsize=10, fontweight='bold', va='top', ha='right', color=clr1)
ax5.text(0.9, 0.55, r'$\mu = 0$', transform=ax5.transAxes, fontsize=10, fontweight='bold', va='top', ha='right', color=clr2)
ax5.text(0.95, 0.15, r'$\mu = -1$', transform=ax5.transAxes, fontsize=10, fontweight='bold', va='top', ha='right', color=clr3)



HM = ax6.imshow(DR, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto', vmin=2, vmax=7)  #vmin=0
ax6.set_xticks([0,   7,   14,  len(Mus)-1])
ax6.set_xticklabels([str(int(min(Mus))), '0', '1' ,str(int(max(Mus)))])
ax6.set_yticks([0, int(1*len(Betas)/4), int(len(Betas)/2), int(3*len(Betas)/4), len(Betas)-1] )
ax6.set_yticklabels([str(int(min(Betas))), str(int(min(Betas)/2)), '0', str(int(max(Betas)/2)), str(int(max(Betas)))])
ax6.axvline(x=7, ls='dashed', color='black')
ax6.axhline(y=int(len(Betas)/2), ls='dashed', color='black')
ax6.set_xlabel(r'$\mu$')
ax6.set_ylabel(r'$\beta$')
fig.colorbar(HM, ax=ax6, label=r'Dynamic range ($\gamma$)')



ax1.text(-0.15, 1.15, "A", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax2.text(-0.15, 1.15, "B", transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax3.text(-0.15, 1.15, "C", transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax4.text(-0.15, 1.15, "D", transform=ax4.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax5.text(-0.15, 1.15, "E", transform=ax5.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax6.text(-0.15, 1.15, "F", transform=ax6.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax1.text(0.9, 0.2, r'$\omega = \Omega_0$', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax2.text(0.9, 0.2, r'$\omega = 1.01\times\Omega_0$', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax3.text(0.9, 0.2, r'$\omega = 1.1\times\Omega_0$', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\Fig5.jpeg', dpi=300)





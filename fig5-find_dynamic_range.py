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






#SLICES TO COMPARE TO ANALYTIC CALCULATION
num_trials    = 100
num_IC_lowF   = 1  #64
num_IC_highF  = 1
dt_lowF  = 0.001*2*np.pi
dt_highF = 0.0001
num_stim_cycles = 5      #5  20
num_cut_cycles  = 5      #5  20
F_min_scaling_factor = 1 #1  25      #incresing the expected F_min so that it can sync in presence of noise. use 25 for mu=1  D=0.001

D  = 0.001
Mus   = np.linspace( -1,  1, 3)
Betas = np.linspace( -9,  9, 10)
delta_w = 0.01
W0, alpha = 1, 1
wf = W0 + delta_w
R_small = 0.01  #0.01
Dyn_Range_an, Dyn_Range_num = np.zeros((len(Betas), len(Mus))), np.zeros((len(Betas), len(Mus)))
for b in range(len(Betas)):
    beta = Betas[b]
    print(b)
    for m in range(len(Mus)):
        mu = Mus[m]
        if mu > 0:
            r0 = (mu/alpha)**0.5
            inc_factor = 10
        else:
            r0 = 0
            inc_factor = 1000
        F_min, F_max = Find_expected_range(mu, delta_w, alpha, beta, inc_factor, R_small=R_small)
        F_min *= F_min_scaling_factor
        rng_analytic = Find_dynamic_range_analytic(mu, delta_w, alpha, beta, inc_factor, R_small=R_small)
        rng_numeric = 0
        R_max = Find_Res_Amp(dt_highF, mu, W0 + beta*r0**2, alpha, beta, D, r0, F_max, wf, num_IC_highF,  num_stim_cycles, num_cut_cycles)  #just doing this long one one time
        for trial in range(num_trials):
            R_min = Find_Res_Amp(dt_lowF,   mu, W0 + beta*r0**2, alpha, beta, D, r0, F_min, wf, num_IC_lowF, num_stim_cycles, num_cut_cycles)
            if mu > 0:
                rng_numeric += np.emath.logn( inc_factor, F_max/F_min  ) / np.emath.logn( inc_factor, R_max/R_min  ) / num_trials
            else:
                rng_numeric += np.emath.logn(10, F_max/F_min)/np.emath.logn(10, R_max/R_min) / num_trials
        #print(F_min, F_max, R_min, R_max, np.round(F_max/F_min, 4), np.round(R_max/R_min, 4))
        print(b, beta, mu, rng_analytic, rng_numeric, np.round(rng_numeric/rng_analytic, 4))
        Dyn_Range_an[b, m], Dyn_Range_num[b, m] = rng_analytic, rng_numeric

plt.figure()
plt.plot(Betas,  Dyn_Range_an[:, 0], color='black')
plt.plot(Betas, Dyn_Range_num[:, 0], "o", color='blue')
plt.plot(Betas,  Dyn_Range_an[:, 1], color='black')
plt.plot(Betas, Dyn_Range_num[:, 1], "o", color='green')
plt.plot(Betas,  Dyn_Range_an[:, 2], color='black')
plt.plot(Betas, Dyn_Range_num[:, 2], "o", color='red')

plt.figure()
plt.imshow(Dyn_Range_num, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto')
plt.figure()
plt.imshow(Dyn_Range_num, cmap='jet', interpolation='none', origin='lower', aspect='auto')







#Combining Heat Maps
#hm1 = np.load(r'C:\Users\Justin\Desktop\fig5_Dyn_Rng_dw=0p01_D=0p001_1IC_5Trial_5cyc_5cut_a.npy')
#hm2 = np.load(r'C:\Users\Justin\Desktop\fig5_Dyn_Rng_dw=0p01_D=0p001_1IC_5Trial_5cyc_5cut_b.npy')
#hm3 = np.load(r'C:\Users\Justin\Desktop\fig5_Dyn_Rng_dw=0p01_D=0p001_1IC_10Trial_5cyc_5cut_c.npy')
#hm_avg = (hm1 + hm2 + 2*hm3)/4
#np.save(r'C:\Users\Justin\Desktop\fig5_Dyn_Rng_dw=0p01_D=0p001_1IC_20Trial_5cyc_5cut', hm_avg )

'''
#FINDING COMPRESSION RANGE HEAT MAP
num_IC = 1  #32+ with noise,  1 without
num_trials = 10
Betas = np.linspace(-10, 10, 21)  #21pts  #For heat maps
Mus   = np.linspace( -1,  2, 22)  #22pts
#Betas = np.linspace(-10, 10, 11) #for making slices
#Mus   = np.linspace( -1,  2, 4)
num_stim_cycles = 5   #10,  5
num_cut_cycles  = 5   #20,  2  #needs to be pretty long, at least 20 for mu=1
delta_w = 0.01
D  = 0.001

dt = 0.001
W0, alpha = 1, 1
wf = W0 + delta_w
R_small = 0.01  #0.01
Dyn_Range_an, Dyn_Range_num = np.zeros((len(Betas), len(Mus))), np.zeros((len(Betas), len(Mus)))
for b in range(len(Betas)):
    for m in range(len(Mus)):
        mu = Mus[m]
        beta = Betas[b]
        dt_tiny = dt/3
        if mu > 0:
            r0 = (mu/alpha)**0.5
            inc_factor = 10
            if mu > 1:
                dt_tiny = dt/10
        else:
            r0 = 0
            inc_factor = 1000
        F_min, F_max = Find_expected_range(mu, delta_w, alpha, beta, inc_factor, R_small=R_small)
        rng_numeric = 0
        for trial in range(num_trials):
            R_min = Find_Res_Amp(dt,      mu, W0 + beta*r0**2, alpha, beta, D, r0, F_min, wf, num_IC, num_stim_cycles, num_cut_cycles)
            R_max = Find_Res_Amp(dt_tiny, mu, W0 + beta*r0**2, alpha, beta, D, r0, F_max, wf, num_IC, num_stim_cycles, num_cut_cycles)
            #print(F_min, F_max, R_min, R_max, np.round(F_max/F_min, 4), np.round(R_max/R_min, 4))
            rng_analytic = Find_dynamic_range_analytic(mu, delta_w, alpha, beta, inc_factor, R_small=R_small)
            if mu > 0:
                rng_numeric += np.emath.logn( inc_factor, F_max/F_min  ) / np.emath.logn( inc_factor, R_max/R_min  ) / num_trials
            else:
                rng_numeric += np.emath.logn(10, F_max/F_min)/np.emath.logn(10, R_max/R_min) / num_trials
        print(b, beta, mu, rng_analytic, rng_numeric, np.round(rng_numeric/rng_analytic, 4))
        Dyn_Range_an[b, m], Dyn_Range_num[b, m] = rng_analytic, rng_numeric
plt.figure()
plt.imshow(Dyn_Range_an, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto')
plt.figure()
plt.imshow(Dyn_Range_an, cmap='jet', interpolation='none', origin='lower', aspect='auto')
plt.figure()
plt.imshow(Dyn_Range_num, cmap='jet', interpolation='bilinear', origin='lower', aspect='auto')
plt.figure()
plt.imshow(Dyn_Range_num, cmap='jet', interpolation='none', origin='lower', aspect='auto')
'''

#np.save(r'C:\Users\Justin\Desktop\fig5_Dyn_Rng_dw=0p01_D=0p001_1IC_10Trial_5cyc_5cut_c', Dyn_Range_num )

'''
plt.figure()  #plotting slices
plt.plot(Betas, Dyn_Range_an[:,0], color='black')
plt.plot(Betas, Dyn_Range_num[:,0], "o", color='blue')
plt.plot(Betas, Dyn_Range_an[:,1], color='black')
plt.plot(Betas, Dyn_Range_num[:,1], "o", color='green')
plt.plot(Betas, Dyn_Range_an[:,2], color='black')
plt.plot(Betas, Dyn_Range_num[:,2], "o", color='red')
plt.plot(Betas, Dyn_Range_an[:,3], color='black')
plt.plot(Betas, Dyn_Range_num[:,3], "o", color='purple')
plt.ylim(0, 1.2*max(Dyn_Range_num[:,3]) )
'''


'''

plt.figure()  #plotting slices
plt.plot(Betas, Dyn_Range_an[:,0], color='black')
plt.plot(Betas, Dyn_Range_num[:,0], "o", color='blue')
plt.plot(Betas, Dyn_Range_an[:,7], color='black')
plt.plot(Betas, Dyn_Range_num[:,7], "o", color='green')
plt.plot(Betas, Dyn_Range_an[:,14], color='black')
plt.plot(Betas, Dyn_Range_num[:,14], "o", color='red')
plt.plot(Betas, Dyn_Range_an[:,21], color='black')
plt.plot(Betas, Dyn_Range_num[:,21], "o", color='purple')
plt.ylim(0, 1.2*max(Dyn_Range_num[:,21]) )

'''

'''

def Find_Res_Amp_Testing(dt, mu, w0, alpha, beta, D, r0, F, wf, num_IC, num_stim_cycles, num_cut_cycles):
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
    print( "Amp =", ( sin_fit(x_avg[N_cut:], dt, wf) + sin_fit(y_avg[N_cut:], dt, wf) ) / 2  )
    plt.figure()
    plt.plot(x_avg)
    plt.plot(y_avg)
    
Num_IC = 1
Mu = 1
r0 = 1
Beta = 10
Find_Res_Amp_Testing(dt, Mu, W0 + Beta*r0**2, alpha, Beta, D, r0, F_min, wf, Num_IC, 20, 20)

'''




'''
#FINDING COMPRESSION RANGE, Short Cut, finding slices and comparing to analytic claculation
D = 0.001
num_IC = 4  #32+ with noise,  1 without
num_trials = 1
dt = 0.001 #*2*np.pi
dt_tiny = dt/5   #dt/3 for mu=0  or dt/5
mu = 1
inc_factor = 10   #10,   1000 for mu <= 0
delta_w = 0.01

W0, alpha = 1, 1
wf = W0 + delta_w
num_stim_cycles = 10  #10
num_cut_cycles = 10   #10
Betas = np.linspace(-10, 10, 11)
Dyn_Range_an, Dyn_Range_num = np.zeros(len(Betas)), np.zeros(len(Betas))
R_small = 0.01  #0.01
r0 = 0
if mu > 0:
    r0 = (mu/alpha)**0.5
for b in range(len(Betas)):
    print(b)
    beta = Betas[b]
    F_min, F_max = Find_expected_range(mu, delta_w, alpha, beta, inc_factor, R_small=R_small)
    rng_numeric = 0
    for trial in range(num_trials):
        R_min = Find_Res_Amp(dt,      mu, W0 + beta*r0**2, alpha, beta, D, r0, F_min, wf, num_IC, num_stim_cycles, num_cut_cycles)
        R_max = Find_Res_Amp(dt_tiny, mu, W0 + beta*r0**2, alpha, beta, D, r0, F_max, wf, num_IC, num_stim_cycles, num_cut_cycles)
        print(F_min, F_max, R_min, R_max, np.round(F_max/F_min, 4), np.round(R_max/R_min, 4))
        rng_analytic = Find_dynamic_range_analytic(mu, delta_w, alpha, beta, inc_factor, R_small=R_small)
        if mu > 0:
            rng_numeric += np.emath.logn( inc_factor, F_max/F_min  ) / np.emath.logn( inc_factor, R_max/R_min  ) / num_trials
        else:
            rng_numeric += np.emath.logn(10, F_max/F_min)/np.emath.logn(10, R_max/R_min) / num_trials
        #print(F_min/(mu*r0), "<< 1")
    print(rng_analytic, rng_numeric)
    Dyn_Range_an[b], Dyn_Range_num[b] = rng_analytic, rng_numeric
plt.figure()
plt.plot(Betas, Dyn_Range_num, "o", color='red')
smooth_Betas = np.linspace(Betas[0], Betas[-1], 1000)
plt.plot(smooth_Betas, Find_dynamic_range_analytic(mu, delta_w, alpha, smooth_Betas, inc_factor, R_small=R_small), color='black')
plt.ylim(0, 10)
'''













'''
#FINDING COMPRESSION RANGE, LONG OLD WAY
def Find_dynamic_range_numeric(F, R, R_start, inc_factor):
    for i in range(len(R)):
        if R[i] > R_start:
            F_min = F[i]
            break
    for i in range(len(R)):
        if R[i] > inc_factor*R_start:
            F_max = F[i]
            break
    return F_min, F_max, np.emath.logn( inc_factor, F_max/F_min  )
    
D = 0.0
num_IC = 1  #32 with noise,  1 without
dt = 0.001 #*2*np.pi
mu = 1
inc_factor = 3  #3
delta_w = 0.1
W0, alpha = 1, 1
r0 = (mu/alpha)**0.5
wf = W0 + delta_w
num_stim_cycles = 10  #10
num_cut_cycles = 10   #10
num_est_points = 10   #10
#Betas = np.linspace(5, 5, 1)  #-10, 10, 13
Betas = np.linspace(-10, 10, 13)
Dyn_Range_an, Dyn_Range_num = np.zeros(len(Betas)), np.zeros(len(Betas))
for b in range(len(Betas)):
    print(b)
    beta = Betas[b]
    F_min_exp, F_max_exp = Find_expected_range(mu, delta_w, alpha, beta, inc_factor)
    FS = np.concatenate(( np.logspace(  np.log10(0.5*F_min_exp), np.log10(2*F_min_exp),   num_est_points),  np.logspace(  np.log10(0.3*F_max_exp), np.log10(1.5*F_max_exp),   num_est_points) ))
    R = np.zeros(len(FS))
    for i in range(len(FS)):
        R[i] = Find_Res_Amp(dt, mu, W0 + beta*mu/alpha, alpha, beta, D, r0, FS[i], wf, num_IC, num_stim_cycles, num_cut_cycles)
    R_start = 0.95*r0  #0.95 r0 for mu=1
    rng_analytic = Find_dynamic_range_analytic(mu, delta_w, alpha, beta, inc_factor)
    F_min, F_max, rng_numeric = Find_dynamic_range_numeric(FS, R, R_start, inc_factor)
    print(F_min/(mu*r0), "<< 1")
    print(rng_analytic, rng_numeric)
    Dyn_Range_an[b], Dyn_Range_num[b] = rng_analytic, rng_numeric
plt.figure()
plt.plot(Betas, Dyn_Range_num, "o", color='red')
smooth_Betas = np.linspace(Betas[0], Betas[-1], 1000)
plt.plot(smooth_Betas, Find_dynamic_range_analytic(mu, delta_w, alpha, smooth_Betas, inc_factor), color='black')
plt.ylim(0, 10)
plt.figure()
plt.plot(FS, R, "o-", color='black')
plt.axhline(y=R_start, ls='dashed', color='blue')
plt.axhline(y=inc_factor*R_start, ls='dashed', color='blue')
plt.xscale("log")
plt.yscale("log")
'''







'''

#FINDING RESPONSE CURVES. This is done in the FIG5 file
dt = 0.001*2*np.pi
mu = 1
alpha = 1
beta1 = 0
beta2 = 2
beta3 = 5
W0 = 1
r0 = (mu/alpha)**0.5
D = 0.0
wf = 1.01
num_IC = 1  #32,  16
num_stim_cycles = 10  #10
num_cut_cycles = 10   #10
Fs = np.logspace(2, 3, 5)  #-3, 2, 50
R1, R2, R3 = np.zeros(len(Fs)), np.zeros(len(Fs)), np.zeros(len(Fs))
for i in range(len(Fs)):
    print(i)
    R1[i] = Find_Res_Amp(dt, mu, W0 + beta1*mu/alpha, alpha, beta1, D, r0, Fs[i], wf, num_IC, num_stim_cycles, num_cut_cycles)
    #R2[i] = Find_Res_Amp(dt, mu, W0 + beta2*mu/alpha, alpha, beta2, D, r0, Fs[i], wf, num_IC, num_stim_cycles, num_cut_cycles)
    #R3[i] = Find_Res_Amp(dt, mu, W0 + beta3*mu/alpha, alpha, beta3, D, r0, Fs[i], wf, num_IC, num_stim_cycles, num_cut_cycles)
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






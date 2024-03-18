import random
import numpy as np
import matplotlib.pyplot as plt


def sin_fit(x, dt, fit_w):  #fixed freq w: returns amp, phase and offset. fit_freq in units rad/unit time
    tList = np.linspace(0.0, 1.0*(len(x)-1)*dt, len(x))
    b = np.matrix(x, dtype='float').T
    rows = [ [np.sin(fit_w*t), np.cos(fit_w*t), 1] for t in tList]
    A = np.matrix(rows, dtype='float')
    (w,residuals,rank,sing_vals) = np.linalg.lstsq(A,b, rcond=0)
    amplitude = np.linalg.norm([w[0,0],w[1,0]],2)
    return amplitude

def generate_sin_stim(num_segs_per_freq, w0, delta_w, num_freqs, dt, F, num_stim_cycles, num_seg_cycles):
    num_segs = num_segs_per_freq*num_freqs
    freq_indexes = np.zeros(num_segs, dtype=int)
    freqs = np.linspace(w0 - delta_w, w0 + delta_w, num_freqs)
    N_seg = int(round((num_seg_cycles * 2*np.pi/(w0*dt))))
    Fx, Fy = np.zeros(N_seg*num_segs), np.zeros(N_seg*num_segs)
    for i in range(num_segs_per_freq):
        f_indx = np.linspace(0, num_freqs - 1, num_freqs, dtype=int)
        random.shuffle(f_indx)
        for j in range(num_freqs):
            counter = i*num_freqs + j
            w = freqs[f_indx[j]]
            freq_indexes[counter] = f_indx[j]
            num_stim_pts = int(num_stim_cycles * 2*np.pi/(w*dt))
            Fx[counter*N_seg:counter*N_seg + num_stim_pts] = F*np.cos(w*np.linspace(0, (num_stim_pts-1)*dt, num_stim_pts) - np.pi/2)
            Fy[counter*N_seg:counter*N_seg + num_stim_pts] = F*np.sin(w*np.linspace(0, (num_stim_pts-1)*dt, num_stim_pts) - np.pi/2)
    return Fx, Fy, freqs, freq_indexes

def sensitivity_analysis(num_segs_per_freq, w0, dt, num_stim_cycles, num_seg_cycles, x, freqs, freq_indexes, return_CHI=0):
    N_seg = int(round((num_seg_cycles * 2*np.pi/(w0*dt))))
    X = np.zeros((len(freqs), N_seg))
    for i in range(len(freq_indexes)):
        X[freq_indexes[i], :] += x[i*N_seg:(i+1)*N_seg]/num_segs_per_freq
    #plt.figure()
    #for i in range(len(freqs)):
        #plt.plot(X[i, :] + i)
    CHI = np.zeros(len(freqs))
    for i in range(len(freqs)):
        CHI[i] = sin_fit(X[i, :int(num_stim_cycles*2*np.pi/(freqs[i]*dt))], dt, freqs[i] )  / F
    if return_CHI == 1:
        return CHI.mean(), CHI
    else:
        return CHI.mean()

def PSD(x, framerate):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    xff = (2.0/(N*framerate))*np.abs(xf[0:int(N/2)])**2
    return f, xff



'''


#Finding Lyapunov Exponent

def Rossler_dots(a, b, c, x, y, z):
    x_dot = -y - z
    y_dot = x + a*y
    z_dot = b + z*(x - c)
    return x_dot, y_dot, z_dot

def Rossler_RK4(a, b, c, x, y, z, dt):
    xk1, yk1, zk1 = Rossler_dots(a, b, c, x, y, z)
    xk2, yk2, zk2 = Rossler_dots(a, b, c, x + xk1*dt/2,  y + yk1*dt/2,  z + zk1*dt/2)
    xk3, yk3, zk3 = Rossler_dots(a, b, c, x + xk2*dt/2,  y + yk2*dt/2,  z + zk2*dt/2)
    xk4, yk4, zk4 = Rossler_dots(a, b, c, x + xk3*dt  ,  y + yk3*dt  ,  z + zk3*dt)
    new_x, new_y, new_z = x + (dt/6)*(xk1 + 2*xk2 + 2*xk3 + xk4), y + (dt/6)*(yk1 + 2*yk2 + 2*yk3 + yk4), z + (dt/6)*(zk1 + 2*zk2 + 2*zk3 + zk4)
    return new_x, new_y, new_z

a = 0.21
rp = 0.01  #0.01 for chaotic
num_IC = 50  #50 or 100
N     = 10000  #10k
N_cut = 10000  #10k
dt = 0.01
b, c = 0.2, 5.7
tt = np.linspace(0, (N-1)*dt, N)
D = np.zeros(N)
for ic in range(num_IC):
    print(ic)
    X, Y, Z = np.zeros(N+N_cut), np.zeros(N+N_cut), np.zeros(N+N_cut)
    Xp, Yp, Zp = np.zeros(N), np.zeros(N), np.zeros(N)
    phi = 2*np.pi*random.random()
    X[0], Y[0], Z[0] = 10*np.cos(phi), 10*np.sin(phi), 0
    for i in range(1, N+N_cut):
        X[i], Y[i], Z[i] = Rossler_RK4(a, b, c, X[i-1], Y[i-1], Z[i-1], dt)
    X, Y, Z = X[N_cut:], Y[N_cut:], Z[N_cut:]
    phi_p = 2*np.pi*random.random() #perturbing
    theta_p = np.pi*random.random()
    xp, yp, zp = rp*np.sin(theta_p)*np.cos(phi_p), rp*np.sin(theta_p)*np.sin(phi_p), rp*np.cos(theta_p)
    Xp[0], Yp[0], Zp[0] = X[0] + xp,   Y[0] + yp,   Z[0] + zp
    for i in range(1, N):
        Xp[i], Yp[i], Zp[i] = Rossler_RK4(a, b, c, Xp[i-1], Yp[i-1], Zp[i-1], dt)
    D += ( ( (X-Xp)**2 + (Y-Yp)**2 + (Z-Zp)**2 )**0.5 ) / num_IC
log_D = np.log(D)
plt.figure()
plt.plot(X, Y)
plt.plot(Xp, Yp)
plt.figure()
plt.plot(D)

fit_start, fit_stop = 0, 4000
slope, intercept = np.polyfit(tt[fit_start:fit_stop], log_D[fit_start:fit_stop], 1)
print(slope)
plt.figure()
plt.plot(tt, log_D)
plt.plot(tt[fit_start:fit_stop], tt[fit_start:fit_stop]*slope + intercept, color='red')

#a = 0.07   lam = -0.040
#a = 0.09   lam = -0.023
#a = 0.1    lam = -0.014
#a = 0.13   lam = -0.005    same for a=0.15
#a = 0.152  lam = -0.002
#a = 0.1545 lam = +0.003
#a = 0.155  lam = +0.015
#a = 0.16   lam = +0.033
#a = 0.18   lam = +0.073
#a = 0.2    lam = +0.101

#a = 0.26   lam = +0.120
#a = 0.28   lam = +0.129
#a = 0.3    lam = +0.151


'''





#finding sensitivity

def Rossler_dots(a, b, c, x, y, z, fx, fy, fz):
    x_dot = -y - z + fx
    y_dot = x + a*y + fy
    z_dot = b + z*(x - c) + fz
    return x_dot, y_dot, z_dot

def Rossler_RK4(a, b, c, x, y, z, fx, fy, fz, dt):
    xk1, yk1, zk1 = Rossler_dots(a, b, c, x, y, z, fx, fy, fz)
    xk2, yk2, zk2 = Rossler_dots(a, b, c, x + xk1*dt/2,  y + yk1*dt/2,  z + zk1*dt/2, fx, fy, fz)
    xk3, yk3, zk3 = Rossler_dots(a, b, c, x + xk2*dt/2,  y + yk2*dt/2,  z + zk2*dt/2, fx, fy, fz)
    xk4, yk4, zk4 = Rossler_dots(a, b, c, x + xk3*dt  ,  y + yk3*dt  ,  z + zk3*dt,   fx, fy, fz)
    new_x, new_y, new_z = x + (dt/6)*(xk1 + 2*xk2 + 2*xk3 + xk4), y + (dt/6)*(yk1 + 2*yk2 + 2*yk3 + yk4), z + (dt/6)*(zk1 + 2*zk2 + 2*zk3 + zk4)
    return new_x, new_y, new_z

def Rossler(a, b, c, dt, Fx, Fy, Fz):
    N = len(Fx)
    X, Y, Z = np.zeros(N), np.zeros(N), np.zeros(N)
    phi = 2*np.pi*random.random()
    X[0], Y[0], Z[0] = 10*np.cos(phi), 10*np.sin(phi), 0
    for i in range(1, N):
        X[i], Y[i], Z[i] = Rossler_RK4(a, b, c, X[i-1], Y[i-1], Z[i-1], Fx[i-1], Fy[i-1], Fz[i-1], dt)
    return X, Y, Z




#As = np.array([ 0.2 ])
#Ws = np.array([1.08])
As    = np.array([ 0.07,   0.09,    0.1,   0.13,  0.152, 0.1545, 0.155,  0.16,  0.18,   0.2,  0.26,  0.28, 0.3])
Lyaps = np.array([-0.04, -0.023, -0.014, -0.005, -0.002,  0.003, 0.015, 0.033, 0.073, 0.101, 0.120, 0.129, 0.151])
Ws = np.linspace(0.95, 1.3, 50)  #0.9 to 1.2
#Ws = np.linspace(1.0, 1.12, 14)
#Ws = np.linspace(0.95, 1.15, 20)

num_trials = 1
num_IC = 100
N = 10000
N_cut = 10000  #10k

f0 = 0.01  #0.1 or 0.01 good
dt = 0.01
b, c = 0.2, 5.7
tt = np.linspace(0, (N+N_cut-1)*dt, N+N_cut)
CHI = np.zeros((len(As), len(Ws)))
for i in range(len(As)):
    a = As[i]
    for j in range(len(Ws)):
        ws = Ws[j]
        Fx = f0*np.cos(ws*tt - np.pi/2)
        Fy = f0*np.sin(ws*tt - np.pi/2)
        Fz = np.zeros(len(Fx))
        for trial in range(num_trials):
            x_avg, y_avg, z_avg = np.zeros(N), np.zeros(N), np.zeros(N)
            for ic in range(num_IC):
                x, y, z = Rossler(a, b, c, dt, Fx, Fy, Fz)
                x, y, z = x[N_cut:], y[N_cut:], z[N_cut:]
                x_avg += x / num_IC
                y_avg += y / num_IC
                z_avg += z / num_IC
            CHI[i, j] += (sin_fit(x_avg, dt, ws) + sin_fit(y_avg, dt, ws)) / (2*f0 * num_trials)
    print(a, CHI[i,:].mean(), max(CHI[i,:]) )

plt.figure()
f, xf = PSD(x_avg, 2*np.pi/dt)
f, yf = PSD(y_avg, 2*np.pi/dt)
f, zf = PSD(z_avg, 2*np.pi/dt)
plt.plot(f, xf)
plt.plot(f, yf)
plt.plot(f, zf)
plt.xlim(0.1, 2)
plt.ylabel("psd")

plt.figure()
plt.plot(x_avg)
plt.plot(y_avg)
plt.plot(z_avg)
plt.xlabel('Time steps')

plt.figure()
for i in range(len(CHI)):
    plt.plot(Ws, CHI[i, :])
plt.xlabel("Stimulus frequency")
plt.ylabel("Chi")


plt.figure()
plt.plot(Lyaps, CHI.mean(axis=1), "o-", color='black')
plt.axvline(x=0)

plt.figure()
plt.plot(Lyaps, CHI.max(axis=1), "o-", color='red')
plt.axvline(x=0)



#np.save(r'C:\Users\Justin\Desktop\rossler_data557', [As, Lyaps, Ws, CHI], allow_pickle="True")



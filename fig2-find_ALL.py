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
    for ic in range(num_IC):
        x, y = HOPF(dt, mu, w0, alpha, beta, D, Fx, Fy, r0, Initial_Phases[ic])
        x_avg += x/num_IC
    return sin_fit(x_avg[N_cut:], dt, wf)/F




#TE Signal functions


def PSD(x, framerate):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    xff = (2.0/(N*framerate))*np.abs(xf[0:int(N/2)])**2
    return f, xff

def LP_filter(x, framerate, cutoff_f):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    for i in range(len(f)):
        if f[i] > cutoff_f:
            xf[i] = 0.0 + 0.0j
            xf[len(xf)-i-1] = 0.0 + 0.0j
    return np.real(np.fft.ifft(xf))

def generate_AM_signal(dt, N=1000000, F_avg=0.1, F_dev=0.01, w=1):   #returns Fx, Fy
    DF = np.random.normal(0, F_dev, N)
    DF = LP_filter(DF, 2*np.pi/dt, 2*w)  #cut frequencies below characteristic w
    DF *= F_dev/DF.std()
    tt = np.linspace(0, (N-1)*dt, N)
    Fx = (F_avg + DF)*np.cos(w*tt - np.pi/2)
    Fy = (F_avg + DF)*np.sin(w*tt - np.pi/2)
    return Fx, Fy

def generate_FM_signal(dt, N=1000000, F=0.1, w=1, w_dev=0.2):   #returns Fx, Fy
    Dw = np.random.normal(0, w_dev, N)
    Dw = LP_filter(Dw, 2*np.pi/dt, 1*w)
    Dw *= w_dev/Dw.std()
    phase = np.zeros(N)
    phase[0] = -np.pi/2
    for i in range(1, len(phase)):
        phase[i] = phase[i-1] + (w + Dw[i])*dt
    return F*np.cos(phase), F*np.sin(phase)





# TE functions


def make_boxes(x_raw, num_bins):  #even sized boxes split from min to max
    x_boxes = []
    #there are xbins^dimension number of possible x-states
    x = x_raw - x_raw.mean()  #making mean=0 and variance=1
    x /= x.var()
    box_width = (max(x)-min(x))/num_bins
    x -= x.min()  #making minimum 0 so that state index can start at 0
    for i in range(len(x)):
        box = int(x[i]/box_width)
        if box == num_bins:  #max value will get binned into a state too high, this takes care of that.
            box -= 1
        x_boxes.append(box)
    return x_boxes

def make_4quantile_boxes(x):  #split based on quantiles (quartiles) 4 boxes
    x_boxes = []
    edge1, edge2, edge3 = np.quantile(x, [0.25, 0.5, 0.75])
    for i in range(len(x)):
        if x[i] < edge1:
            box = 0
        elif x[i] >= edge3:
            box = 3
        elif x[i] >= edge2:
            box = 2
        else:
            box = 1
        x_boxes.append(box)
    return x_boxes


def make_digital_boxes(x):  # 0 or 1 based on below or above mean
    x_boxes = np.zeros(len(x), dtype=int)
    x_avg = np.mean(x)
    for i in range(len(x)):
        if x[i] > x_avg:
            x_boxes[i] = 1
    return x_boxes

def make_states(boxes, dim, tau):  # max 9 dimensions
    states = []
    for i in range(len(boxes) - dim*tau + 1):
        word = ''
        for j in range(dim):
            word += str(boxes[i + tau*j])
        states.append(word)
    return states

def indexer(x_states):  # takes states like '012302' and renames them integers 0, 1, 2, 3...
    new_states = []
    sorted_states = np.sort(x_states)
    unique_states = []
    unique_states.append(sorted_states[0])
    for i in range(1, len(sorted_states)):
        if sorted_states[i] != sorted_states[i - 1]:
            unique_states.append(sorted_states[i])
    num_unique_states = len(unique_states)
    for i in range(len(x_states)):
        for j in range(num_unique_states):
            if x_states[i] == unique_states[j]:
                new_states.append(j)
                break
    return new_states

def T_Entropy(x_states_plus1, y_states):  # if dim=5, this uses first 4 dims to find probability of getting the 5th.  infromation transfered from y to x
    y_states = y_states[:len(x_states_plus1)]  # cut off last few to make len x and y states match
    next_x = []
    x_states = []
    for i in range(len(x_states_plus1)):
        word = x_states_plus1[i]
        next_x.append(word[-1])
        x_states.append(word[:-1])

    new_x_states = indexer(x_states)  # re naming states to integers 0, 1, 2, 3...
    new_y_states = indexer(y_states)
    new_x_next_states = indexer(next_x)

    num_x_states = max(new_x_states) + 1
    num_y_states = max(new_y_states) + 1
    num_x_next_states = max(new_x_next_states) + 1
    # print num_x_states, num_y_states, num_x_next_states
    assert len(new_x_states) == len(new_y_states)

    Big_P = np.zeros((num_x_states, num_y_states, num_x_next_states), dtype=float)
    P_x_next_condit_X = np.zeros((num_x_states, num_x_next_states), dtype=float)
    P_x_next_condit_XY = np.zeros((num_x_states, num_y_states, num_x_next_states), dtype=float)

    for i in range(len(new_x_states)):
        Big_P[new_x_states[i]][new_y_states[i]][new_x_next_states[i]] += 1.0
        P_x_next_condit_X[new_x_states[i]][new_x_next_states[i]] += 1.0
        P_x_next_condit_XY[new_x_states[i]][new_y_states[i]][new_x_next_states[i]] += 1.0

    Big_P /= np.sum(Big_P)  # Normalizing
    for i in range(num_x_states):
        if np.sum(P_x_next_condit_X[i]) > 0.0:
            P_x_next_condit_X[i] /= np.sum(P_x_next_condit_X[i])  # Normalizing based on conditional info
    for i in range(num_x_states):
        for j in range(num_y_states):
            if np.sum(P_x_next_condit_XY[i][j]) > 0.0:
                P_x_next_condit_XY[i][j] /= np.sum(P_x_next_condit_XY[i][j])  # Normalizing based on conditional info
    T_entropy = 0.0
    for i in range(num_x_states):
        for j in range(num_y_states):
            for k in range(num_x_next_states):
                if P_x_next_condit_XY[i][j][k]*P_x_next_condit_X[i][k] > 0.0:
                    T_entropy += Big_P[i][j][k]*( np.log2(P_x_next_condit_XY[i][j][k]) - np.log2(P_x_next_condit_X[i][k]) )
    return T_entropy  # Big_P, P_x_next_condit_X, P_x_next_condit_XY, new_x_states, new_y_states, new_x_next_states


def Transfer_Entropy(x, y, dim=5, tau=200, box_type='4even'):  # info tranfered from y to x
    if box_type == '4even':
        x_boxes = make_boxes(x, 4)
        y_boxes = make_boxes(y, 4)
    if box_type == '4quantile':
        x_boxes = make_4quantile_boxes(x)
        y_boxes = make_4quantile_boxes(y)
    if box_type == 'digital':
        x_boxes = make_digital_boxes(x)
        y_boxes = make_digital_boxes(y)
    x_states_plus1 = make_states(x_boxes, dim + 1, tau)  # 1 extra dimension to be used in the prediction
    y_states = make_states(y_boxes, dim, tau)
    T = T_Entropy(x_states_plus1, y_states)
    return T

def find_Transfer_Entropy(mu, beta, W0, alpha, D, dt, Fx, Fy, calc_type, Box_type):
    if mu > 0:
        r0 = (mu/alpha)**0.5
    else:
        r0 = 0
    w0 = W0 + beta*(r0**2)
    x, y = HOPF(dt, mu, w0, alpha, beta, D, Fx, Fy, r0, 2*np.pi*random.random())
    if calc_type == 'xy':
        te =  (Transfer_Entropy(x, Fx, dim=5, tau=200, box_type=Box_type)  +   Transfer_Entropy(y, Fy, dim=5, tau=200, box_type=Box_type)) / 2     #averaging x, y components
        te_rev =  (Transfer_Entropy(Fx, x, dim=5, tau=200, box_type=Box_type)  +   Transfer_Entropy(Fy, y, dim=5, tau=200, box_type=Box_type)) / 2
    if calc_type == 'radius':
        r = (x**2 + y**2)**0.5
        Fr = (Fx**2 + Fy**2)**0.5
        te = Transfer_Entropy(r, Fr, dim=5, tau=200, box_type=Box_type)   #tau = 200 standard
        te_rev = Transfer_Entropy(Fr, r, dim=5, tau=200, box_type=Box_type)
    return te, te_rev





'''
#GENERATING NOISE SWEEP FOR CHI   Need to correct Chi inputs w0, r0
dt = 0.001*2*np.pi  #Keep Fixed
W0 = 1
wf = W0
F = 0.01
alpha = 1
mu1, mu2 = 0, 1
beta1, beta2 = 0, 5
r0_1, r0_2 = 0, 1
w0_1, w0_2 = W0 + beta1*mu1/alpha, W0 + beta2*mu2/alpha
Ds = np.logspace(-5, 0, 13)
Ds = np.insert(Ds, 0, 0.0)
Chi1 = np.zeros(len(Ds))
Chi2 = np.zeros(len(Ds))
for d in range(len(Ds)):
    print(d)
    D = Ds[d]
    Chi1[d] = Find_Chi(dt, mu1, w0_1, alpha, beta1, D, r0_1, F=0.01, wf=1, num_IC=128, num_stim_cycles=20, num_cut_cycles=20)
    Chi2[d] = Find_Chi(dt, mu2, w0_2, alpha, beta2, D, r0_2, F=0.01, wf=1, num_IC=128, num_stim_cycles=20, num_cut_cycles=20)
Chi1 = np.round(Chi1, 2)
Chi2 = np.round(Chi2, 2)
plt.figure()
plt.plot(Ds[1:], Chi1[1:], "o-", color='black')
plt.plot(Ds[1:], Chi2[1:], "o-", color='red')
plt.axhline(y=Chi1[0], color='black')
plt.axhline(y=Chi2[0], color='red')
plt.xscale("log")
'''


'''
#GENERATING HEAT MAPS FOR CHI
dt = 0.001*2*np.pi  #Keep Fixed
D = 0.001
W0 = 1
alpha = 1
Mus    = np.linspace(-1, 2, 22)  #4, 7, 10  + 3n,  22 good
Betas = np.linspace(-10, 10, 21)  #21
CHI = np.zeros((len(Betas), len(Mus)))
for b in range(len(Betas)):
    print(b)
    for m in range(len(Mus)):
        mu, beta = Mus[m], Betas[b]
        CHI[b, m] = Find_Chi(dt, mu, W0, alpha, beta, D, F=0.01, wf=W0, num_IC=64, num_stim_cycles=20, num_cut_cycles=20)
plt.figure()
plt.imshow(CHI, cmap='bwr', interpolation='nearest', origin='lower', aspect='auto', vmin=0)  #vmin=0
plt.xticks([0, int(len(Mus)/3),   len(Mus)-1],   [str(min(Mus)), '0' ,str(max(Mus))] )
plt.yticks([0, int(len(Betas)/2), len(Betas)-1], [str(min(Betas)), '0' ,str(max(Betas))] )
plt.axvline(x=int(len(Mus)/3), ls='dashed', color='black')
plt.axhline(y=int(len(Betas)/2), ls='dashed', color='black')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\beta$')
plt.colorbar(label='Chi')
'''
#np.save(r'C:\Users\Justin\Desktop\CHI_D=0p001_64IC', CHI )





'''
#NOISE SWEEP for TE both AM (digital), and FM (digital), calc_type=xy
dt = 0.001*2*np.pi  #Keep Fixed
W0 = 1
alpha = 1
mu1,     mu2 = 0, 1
beta1, beta2 = 0, 5
calc_type = 'xy'  #'xy' or 'radius'
Box_type = 'digital'    #'4even' '4quantile' 'digital'
Fx, Fy = generate_AM_signal(dt, N=1000000, F_avg=0.0, F_dev=0.5, w=W0)
#Fx, Fy = generate_FM_signal(dt, N=100000,     F=0.1,  w=W0,  w_dev=0.3)
Ds = np.logspace(-5, 0, 13)   #13 steps
Ds = np.insert(Ds, 0, 0.0)
TE1, TE2, TE1_rev, TE2_rev = np.zeros(len(Ds)), np.zeros(len(Ds)), np.zeros(len(Ds)), np.zeros(len(Ds))
for d in range(len(Ds)):
    print(d)
    D = Ds[d]
    TE1[d], TE1_rev[d] = find_Transfer_Entropy(mu1, beta1, W0, alpha, D, dt, Fx, Fy, calc_type, Box_type)
    TE2[d], TE2_rev[d] = find_Transfer_Entropy(mu2, beta2, W0, alpha, D, dt, Fx, Fy, calc_type, Box_type)
TE1, TE2, TE1_rev, TE2_rev = np.round(TE1, 4), np.round(TE2, 4), np.round(TE1_rev, 4), np.round(TE2_rev, 4)
plt.figure()
plt.plot(Ds[1:], TE1[1:], "o-", color='black')
plt.plot(Ds[1:], TE2[1:], "o-", color='red')
plt.plot(Ds[1:], TE1_rev[1:], "--", color='black')
plt.plot(Ds[1:], TE2_rev[1:], "--", color='red')
plt.axhline(y=TE1[0], color='black')
plt.axhline(y=TE2[0], color='red')
plt.xscale("log")
'''



#GENERATING TE HEAT MAPS
D = 0.001
dt = 0.001*2*np.pi  #Keep Fixed
W0 = 1
alpha = 1
calc_type = 'xy'  #'xy' or 'radius'
Box_type = 'digital'    #'4even' '4quantile' 'digital'
#Fx, Fy = generate_AM_signal(dt, N=1000000, F_avg=0.0, F_dev=0.5, w=W0)
Fx, Fy = generate_FM_signal(dt,  N=1000000,     F=0.1,  w=W0,  w_dev=0.3)
Mus    = np.linspace(-1, 2, 22)  #4, 7, 10  + 3n,  22 good
Betas = np.linspace(-10, 10, 21)  #21
TE = np.zeros((len(Betas), len(Mus)))
TE_rev = np.zeros((len(Betas), len(Mus)))
for b in range(len(Betas)):
    print(b)
    for m in range(len(Mus)):
        mu, beta = Mus[m], Betas[b]
        TE[b, m], TE_rev[b, m]= find_Transfer_Entropy(mu, beta, W0, alpha, D, dt, Fx, Fy, calc_type, Box_type)
plt.figure()
plt.imshow(TE, cmap='bwr', interpolation='nearest', origin='lower', aspect='auto')  #vmin=0
plt.xticks([0, int(len(Mus)/3),   len(Mus)-1],   [str(min(Mus)), '0' ,str(max(Mus))] )
plt.yticks([0, int(len(Betas)/2), len(Betas)-1], [str(min(Betas)), '0' ,str(max(Betas))] )
plt.axvline(x=int(len(Mus)/3), ls='dashed', color='black')
plt.axhline(y=int(len(Betas)/2), ls='dashed', color='black')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\beta$')
plt.colorbar(label='Transfer entropy (bits)')
plt.figure()
plt.imshow(TE_rev, cmap='bwr', interpolation='nearest', origin='lower', aspect='auto')  #vmin=0
plt.xticks([0, int(len(Mus)/3),   len(Mus)-1],   [str(min(Mus)), '0' ,str(max(Mus))] )
plt.yticks([0, int(len(Betas)/2), len(Betas)-1], [str(min(Betas)), '0' ,str(max(Betas))] )
plt.axvline(x=int(len(Mus)/3), ls='dashed', color='black')
plt.axhline(y=int(len(Betas)/2), ls='dashed', color='black')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\beta$')
plt.colorbar(label='Reverse transfer entropy (bits)')

np.save(r'C:\Users\Justin\Desktop\TE_FM_D=0p001_beta10_mu2_N=1M', [TE, TE_rev] )




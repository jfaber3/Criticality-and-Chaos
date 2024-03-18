import random
import numpy as np
import matplotlib.pyplot as plt


#narrow freq range, 100ic 4 trials f0=0.01
As2 = np.array([0.07  , 0.09  , 0.1   , 0.13  , 0.152 , 0.1545, 0.155 , 0.16 , 0.18  , 0.2   , 0.26  , 0.28  , 0.3   ])
Lyaps2 = np.array([-0.04 , -0.023, -0.014, -0.005, -0.002,  0.003,  0.015,  0.033, 0.073,  0.101,  0.12 ,  0.129,  0.151])
Ws2 = np.array([1.  , 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12])
CHI2 = np.array([[117.95400978,  79.79544147,  65.98256301,  84.37188009,
        103.40895514,  54.05327318,  40.6905526 ,  26.87978837,
         16.76290002,  10.55347358,   5.44815529,  19.17593713,
         21.85606421],
       [ 91.21315522, 167.97595086, 142.47661488,  66.03831304,
         58.58438034,  93.77707491,  66.8612643 ,  59.2146791 ,
         19.65324056,   8.68570744,  10.21766191,   9.44012683,
         27.30802728],
       [103.78657323, 211.15677072, 177.3500885 , 121.38423   ,
         78.45899974,  76.56029235, 160.93042148, 101.26851279,
         66.50885714,   8.32062689,  13.63352122,  17.67233328,
         23.00592135],
       [103.93041651, 190.45377576, 305.02274648, 392.459784  ,
        374.05867432, 328.29797076, 239.11273529, 234.66765508,
        192.65875642, 130.97124352,  46.54378837,  14.53882047,
         44.11407608],
       [ 82.57031463, 147.01815641, 258.27997723, 396.81582675,
        462.68695808, 494.5222465 , 423.99050825, 326.83129425,
        274.04851124, 228.20948764, 145.22772452,  39.3186162 ,
         33.46627266],
       [ 79.29110119, 151.29209333, 232.39438651, 381.16476228,
        469.59143464, 505.14199039, 414.86090022, 356.51804803,
        282.13082632, 238.31494506, 157.38825447,  50.10891796,
         30.81896653],
       [ 78.09201598, 142.01553069, 278.95427692, 376.70548043,
        468.88467464, 488.72575481, 425.23249786, 375.30537805,
        306.19785872, 263.00261058, 159.90059957,  52.26704805,
         29.35572439],
       [ 68.23982764, 134.63099759, 253.16420736, 358.78880927,
        486.86795953, 503.92824246, 473.15723345, 377.44993785,
        301.91377117, 275.19347944, 188.01237896,  75.13172152,
         16.40740251],
       [  6.39047775,  90.07673987, 183.73022525, 297.5303478 ,
        446.23751758, 519.17886353, 511.62691114, 501.68897054,
        414.5380408 , 341.11224279, 258.80905629, 157.92389379,
         53.24741876],
       [ 62.95944821,  20.54062345,  87.92896242, 156.70264026,
        269.3120286 , 399.12788183, 494.39665409, 524.62356237,
        478.67293142, 395.2207043 , 319.709764  , 247.15861909,
        165.39188689],
       [ 26.4100802 ,  30.37808217,  26.67415857,  39.30663226,
         28.79371504,  13.47046404,   7.84986808,  24.08317105,
         38.22965138,  35.25155047,  41.2082688 ,  46.20176861,
         48.44414234],
       [ 18.6668894 ,  25.64674315,  21.3017773 ,  28.84717362,
         35.85075185,  39.86782917,  31.68192901,  44.29605375,
         37.370883  ,  24.92530132,  32.15623246,  26.73306286,
         22.43780841],
       [ 30.35600659,  28.55586946,  37.71604156,  32.34215169,
         38.55969107,  27.51622526,  33.81121527,  25.03516424,
         31.45992094,  30.23220994,  20.29556959,  24.75434191,
         15.34584668]])



As, Lyaps, Ws, CHI = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig_s5_Rossler_data.npy' , allow_pickle="True")
I1, I2, I3, I4, I5 = 0, 3, 5, 8, 12
#Lyaps = array([-0.04 , -0.023, -0.014, -0.005, -0.002,  0.003,  0.015,  0.033, 0.073,  0.101,  0.111,  0.129,  0.151])


#Making the Traces
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

def move_avg(x, window):
    if window%2 != 1:
        print("window must be odd")
        window += 1
    filt_x = np.zeros(len(x), dtype=float)
    for i in range(int(window/2),  len(x) - int(window/2)):  # filter middle of data
        filt_x[i] = sum(x[i - int(window/2):i + int(window/2) + 1])/window
    for i in range(int(window/2)):  # filter ends of data
        filt_x[i] = sum(x[0:(i + int(window/2))]) / len(x[0:(i + int(window/2))])
        filt_x[len(x) - 1 - i] = sum(x[(len(x) - i - int(window/2)):len(x)]) / len(x[(len(x) - i - int(window/2)):len(x)])
    return filt_x


dt = 0.01
N = 20000
N_cut = 20000
b, c = 0.2, 5.7
x1, y1, z1 = np.zeros(N+N_cut), np.zeros(N+N_cut), np.zeros(N+N_cut)
x2, y2, z2 = np.zeros(N+N_cut), np.zeros(N+N_cut), np.zeros(N+N_cut)
x3, y3, z3 = np.zeros(N+N_cut), np.zeros(N+N_cut), np.zeros(N+N_cut)
x4, y4, z4 = np.zeros(N+N_cut), np.zeros(N+N_cut), np.zeros(N+N_cut)
x5, y5, z5 = np.zeros(N+N_cut), np.zeros(N+N_cut), np.zeros(N+N_cut)
a1, a2, a3, a4, a5 = As[I1], As[I2], As[I3], As[I4], As[I5]
x1[0], x2[0], x3[0], x4[0], x5[0] = 10, 10, 10, 10, 10
for i in range(1, N + N_cut):
    x1[i], y1[i], z1[i] = Rossler_RK4(a1, b, c, x1[i - 1], y1[i - 1], z1[i - 1], dt)
    x2[i], y2[i], z2[i] = Rossler_RK4(a2, b, c, x2[i - 1], y2[i - 1], z2[i - 1], dt)
    x3[i], y3[i], z3[i] = Rossler_RK4(a3, b, c, x3[i - 1], y3[i - 1], z3[i - 1], dt)
    x4[i], y4[i], z4[i] = Rossler_RK4(a4, b, c, x4[i - 1], y4[i - 1], z4[i - 1], dt)
    x5[i], y5[i], z5[i] = Rossler_RK4(a5, b, c, x5[i - 1], y5[i - 1], z5[i - 1], dt)
x1, y1, z1 = x1[N_cut:], y1[N_cut:], z1[N_cut:]
x2, y2, z2 = x2[N_cut:], y2[N_cut:], z2[N_cut:]
x3, y3, z3 = x3[N_cut:], y3[N_cut:], z3[N_cut:]
x4, y4, z4 = x4[N_cut:], y4[N_cut:], z4[N_cut:]
x5, y5, z5 = x5[N_cut:], y5[N_cut:], z5[N_cut:]




fig = plt.figure(figsize=(10, 5))
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.93, wspace=5, hspace=2)
ax1  = plt.subplot2grid((5, 30), (0,0),  colspan=5, rowspan=2)
ax2  = plt.subplot2grid((5, 30), (0,6),  colspan=5, rowspan=2)
ax3  = plt.subplot2grid((5, 30), (0,12),  colspan=5, rowspan=2)
ax4  = plt.subplot2grid((5, 30), (0,18),  colspan=5, rowspan=2)
ax5  = plt.subplot2grid((5, 30), (0,24), colspan=5, rowspan=2)
ax6  = plt.subplot2grid((5, 30), (2,0),  colspan=8, rowspan=3)
ax7  = plt.subplot2grid((5, 30), (2,10),  colspan=8, rowspan=3)
ax8  = plt.subplot2grid((5, 30), (2,20), colspan=10, rowspan=3)

ln_width = 0.2
ax1.plot(x1, y1, color='black', linewidth=ln_width)
ax2.plot(x2, y2, color='black', linewidth=ln_width)
ax3.plot(x3, y3, color='black', linewidth=ln_width)
ax4.plot(x4, y4, color='black', linewidth=ln_width)
ax5.plot(x5, y5, color='black', linewidth=ln_width)

lim = 16
ax1.set_xlim(-lim, lim), ax1.set_ylim(-lim, lim)
ax2.set_xlim(-lim, lim), ax2.set_ylim(-lim, lim)
ax3.set_xlim(-lim, lim), ax3.set_ylim(-lim, lim)
ax4.set_xlim(-lim, lim), ax4.set_ylim(-lim, lim)
ax5.set_xlim(-lim, lim), ax5.set_ylim(-lim, lim)

ax1.set_xticks([]), ax1.set_yticks([])
ax2.set_xticks([]), ax2.set_yticks([])
ax3.set_xticks([]), ax3.set_yticks([])
ax4.set_xticks([]), ax4.set_yticks([])
ax5.set_xticks([]), ax5.set_yticks([])

ax1.text(0.7, 1.2, r'$\lambda \approx$' + str(np.round(Lyaps[I1], 2)),  transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax2.text(0.7, 1.2, r'$\lambda \approx$' + str(np.round(Lyaps[I2], 3)),  transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax3.text(0.7, 1.2, r'$\lambda \approx$' + str(np.round(Lyaps[I3], 3)),  transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax4.text(0.7, 1.2, r'$\lambda \approx$' + str(np.round(Lyaps[I4], 2)),  transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax5.text(0.7, 1.2, r'$\lambda \approx$' + str(np.round(Lyaps[I5], 2)),  transform=ax5.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

ax1.set_xlabel('x(t)')
ax1.set_ylabel('y(t)')
ax2.set_xlabel('x(t)')
ax2.set_ylabel('y(t)')
ax3.set_xlabel('x(t)')
ax3.set_ylabel('y(t)')
ax4.set_xlabel('x(t)')
ax4.set_ylabel('y(t)')
ax5.set_xlabel('x(t)')
ax5.set_ylabel('y(t)')

ax6.plot(Ws, move_avg(CHI[I1, :], 3), color='blue', label=r'$\lambda \approx$' + str(np.round(Lyaps[I1], 2)))
ax6.plot(Ws, move_avg(CHI[I2, :], 3), color='springgreen', label=r'$\lambda \approx$' + str(np.round(Lyaps[I2], 3)))
ax6.plot(Ws, move_avg(CHI[I3, :], 3), color='darkorange', label=r'$\lambda \approx$' + str(np.round(Lyaps[I3], 3)))
ax6.plot(Ws, move_avg(CHI[I4, :], 3), color='red', label=r'$\lambda \approx$' + str(np.round(Lyaps[I4], 2)))
ax6.plot(Ws, move_avg(CHI[I5, :], 3), color='black', label=r'$\lambda \approx$' + str(np.round(Lyaps[I5], 2)))
ax6.set_xlabel(r'Stimulus frequency ($\omega$)' )
ax6.set_ylabel(r'$\chi(\omega)$')
ax6.set_ylim(0, 600)
ax6.set_xlim(0.93, 1.4)
ax6.legend(fontsize=9)

ax7.plot(As, CHI.mean(axis=1), "s:", color='black')
ax7.plot(As, CHI.max(axis=1), "o:", color='darkviolet')
ax7.set_xlabel(r'System parameter ($a$)' )
ax7.set_ylim(0, 600)
ax7.set_ylabel(r'$\chi(\omega)$')


ax8.plot(Lyaps, CHI.mean(axis=1), "s:", color='black')
ax8.plot(Lyaps, CHI.max(axis=1), "o:", color='darkviolet')
ax8.set_xlabel(r'Lyapunov exponent ($\lambda)$' )
ax8.set_ylim(0, 600)
ax8.set_xlim(-0.05, 0.16)
ax8.set_ylabel(r'$\chi(\omega)$')
ax8.axvline(x=0, ls='dashed', color='black')

ax1.text(-0.25, 1.2, "A", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax6.text(-0.15, 1.15, "B", transform=ax6.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax7.text(-0.15, 1.15, "C", transform=ax7.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax8.text(-0.15, 1.15, "D", transform=ax8.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\FigS5.jpeg', dpi=300)






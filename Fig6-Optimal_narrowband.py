import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage   #for conour lines

LYAP              = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig1_LYAP_beta10_mu2_0p5cycle_300ic.npy')
CHI               = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig2_CHI_D=0p001_64IC.npy' )
TE_fm, TE_fm_rev  = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig6_TE_FM_D=0p001_beta10_mu2_N=1M.npy' )
Q, TBW            = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig4_heatmaps_beta10_mu2_60IC.npy' , allow_pickle="True")
TAU               = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig3_Tau_off_heatmap_beta10_mu2_64ic_10trials.npy' )
GAMMA             = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig5_Dyn_Rng_dw=0p01_D=0p001_1IC_20Trial_5cyc_5cut.npy' )

LYAP = gaussian_filter(LYAP, 1)  #sigma
CHI = gaussian_filter(CHI, 1)
TE_fm = gaussian_filter(TE_fm, 1)
Q = gaussian_filter(Q, 1)
TAU = gaussian_filter(TAU, 2)
SPEED = 1/TAU
GAMMA = gaussian_filter(GAMMA, 1)

#Creating Lyaponov slices at constant MU for all 5 measures
length = 11
mu_lyap_index1 = 8   #mu11 about 0.57       mu8 about 0.14
Lyap_slice1    = LYAP[:, mu_lyap_index1].copy()
L_Chi_slice1   = CHI[:, mu_lyap_index1].copy()
L_Q_slice1     =   Q[:, mu_lyap_index1].copy()
L_TE_fm_slice1 =  TE_fm[:, mu_lyap_index1].copy()
L_Speed_slice1 =  SPEED[:, mu_lyap_index1].copy()
L_Gamma_slice1 =  GAMMA[:, mu_lyap_index1].copy()
Lyap_slice1    = (   Lyap_slice1[::-1][length-1:] + Lyap_slice1[-length:])/2    #colapsing both sides onto one
L_Chi_slice1   = (  L_Chi_slice1[::-1][length-1:] + L_Chi_slice1[-length:])/2
L_Q_slice1     = (    L_Q_slice1[::-1][length-1:] + L_Q_slice1[-length:])/2
L_TE_fm_slice1 = (L_TE_fm_slice1[::-1][length-1:] + L_TE_fm_slice1[-length:])/2
L_Speed_slice1 = (L_Speed_slice1[::-1][length-1:] + L_Speed_slice1[-length:])/2
L_Gamma_slice1 = (L_Gamma_slice1[::-1][length-1:] + L_Gamma_slice1[-length:])/2
L_Chi_slice1   -= min(L_Chi_slice1)
L_Q_slice1     -= min(L_Q_slice1)
L_TE_fm_slice1 -= min(L_TE_fm_slice1)
L_Speed_slice1 -= min(L_Speed_slice1)
L_Gamma_slice1 -= min(L_Gamma_slice1)
L_Chi_slice1   /= max(L_Chi_slice1)
L_Q_slice1     /= max(L_Q_slice1)
L_TE_fm_slice1 /= max(L_TE_fm_slice1)
L_Speed_slice1 /= max(L_Speed_slice1)
L_Gamma_slice1 /= max(L_Gamma_slice1)


mu_lyap_index2 = 18  #mu18 about 1.57  mu14=1  #mu16=1.2857
Lyap_slice2    = LYAP[:, mu_lyap_index2].copy()
L_Chi_slice2   = CHI[:, mu_lyap_index2].copy()
L_Q_slice2     =   Q[:, mu_lyap_index2].copy()
L_TE_fm_slice2 =  TE_fm[:, mu_lyap_index2].copy()
L_Speed_slice2 =  SPEED[:, mu_lyap_index2].copy()
L_Gamma_slice2 =  GAMMA[:, mu_lyap_index2].copy()
Lyap_slice2    = (   Lyap_slice2[::-1][length-1:] + Lyap_slice2[-length:])/2    #colapsing both sides onto one
L_Chi_slice2   = (  L_Chi_slice2[::-1][length-1:] + L_Chi_slice2[-length:])/2
L_Q_slice2     = (    L_Q_slice2[::-1][length-1:] + L_Q_slice2[-length:])/2
L_TE_fm_slice2 = (L_TE_fm_slice2[::-1][length-1:] + L_TE_fm_slice2[-length:])/2
L_Speed_slice2 = (L_Speed_slice2[::-1][length-1:] + L_Speed_slice2[-length:])/2
L_Gamma_slice2 = (L_Gamma_slice2[::-1][length-1:] + L_Gamma_slice2[-length:])/2
L_Chi_slice2   -= min(L_Chi_slice2)
L_Q_slice2     -= min(L_Q_slice2)
L_TE_fm_slice2 -= min(L_TE_fm_slice2)
L_Speed_slice2 -= min(L_Speed_slice2)
L_Gamma_slice2 -= min(L_Gamma_slice2)
L_Chi_slice2   /= max(L_Chi_slice2)
L_Q_slice2     /= max(L_Q_slice2)
L_TE_fm_slice2 /= max(L_TE_fm_slice2)
L_Speed_slice2 /= max(L_Speed_slice2)
L_Gamma_slice2 /= max(L_Gamma_slice2)



CHI -= CHI.min()
TE_fm  -= TE_fm.min()
Q   -= Q.min()
SPEED_min = SPEED.min()
SPEED -= SPEED_min
GAMMA -= GAMMA.min()

quantile = 0.9
Chi_thresh   = np.quantile(CHI,   quantile)
TE_fm_thresh = np.quantile(TE_fm, quantile)
Q_thresh     = np.quantile(Q,     quantile)
Speed_thresh = np.quantile(SPEED, quantile)
Gamma_thresh = np.quantile(GAMMA, quantile)

CHI   /= Chi_thresh
TE_fm /= TE_fm_thresh
Q     /= Q_thresh
SPEED /= Speed_thresh
GAMMA /= Gamma_thresh
#rectifying at 1
CHI[CHI > 1] = 1
TE_fm[TE_fm > 1] = 1
Q[Q > 1] = 1
SPEED[SPEED > 1] = 1
GAMMA[GAMMA > 1] = 1

Mus   = np.linspace(-1, 2, 22)
Betas = np.linspace(-10, 10, 21)
color_map = 'jet'  #jet bwr
fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.93, wspace=3, hspace=0.5)
ax1  = plt.subplot2grid((3, 15), (0,0), colspan=3, rowspan=1)
ax2  = plt.subplot2grid((3, 15), (0,3), colspan=3, rowspan=1)
ax3  = plt.subplot2grid((3, 15), (0,6), colspan=3, rowspan=1)
ax4  = plt.subplot2grid((3, 15), (0,9), colspan=3, rowspan=1)
ax5  = plt.subplot2grid((3, 15), (0,12), colspan=3, rowspan=1)

ax10  = plt.subplot2grid((3, 15), (1,5), colspan=7, rowspan=2)

ax6  = plt.subplot2grid((3, 15), (1,12), colspan=3, rowspan=1)
ax7  = plt.subplot2grid((3, 15), (2,12), colspan=3, rowspan=1)

ax8  = plt.subplot2grid((3, 15), (1,0), colspan=3, rowspan=1)
ax9  = plt.subplot2grid((3, 15), (2,0), colspan=3, rowspan=1)



ax1.imshow(CHI, cmap=color_map, interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=1)
ax2.imshow(TE_fm, cmap=color_map, interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=1)
ax3.imshow(Q, cmap=color_map, interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=1)
ax4.imshow(SPEED, cmap=color_map, interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=1)
ax5.imshow(GAMMA, cmap=color_map, interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=1)


interpolation_factor = 3
beta_slice_color1, beta_slice_color2, beta_slice_color3, beta_slice_color4 = 'mediumblue', 'blue', 'deepskyblue', 'cyan'
mu_slice_color1 = 'magenta'
SUM1 = 1*CHI  +  1*TE_fm  +  1*Q  +  1*SPEED   + 1*GAMMA
SUM1 = gaussian_filter(SUM1, 0.1)
SUM1 = scipy.ndimage.zoom(SUM1, interpolation_factor)  #3 because we interpolated
Mus_big, Betas_big = np.linspace(min(Mus), max(Mus), interpolation_factor*len(Mus)), np.linspace(min(Betas), max(Betas), interpolation_factor*len(Betas))
SUM1 = gaussian_filter(SUM1, 1)
SUM1 /= SUM1.max()  #normalizing
beta_slice_mu_index1 = 12 * interpolation_factor
beta_slice_mu_index2 = 14 * interpolation_factor
beta_slice_mu_index3 = 17 * interpolation_factor
beta_slice_mu_index4 = 21 * interpolation_factor
mu_slice_beta_index1 = 13 * interpolation_factor
beta_slice1 = SUM1[:, beta_slice_mu_index1]
beta_slice2 = SUM1[:, beta_slice_mu_index2]
beta_slice3 = SUM1[:, beta_slice_mu_index3]
beta_slice4 = SUM1[:, beta_slice_mu_index4]
mu_slice1 = SUM1[mu_slice_beta_index1, :]

HM = ax10.contourf(SUM1, cmap=color_map, levels=np.linspace(SUM1.min(), SUM1.max(), 200) )   #80 contours good
ax10.set_xticks([0, int(len(SUM1[0, :])/3), int(2*len(SUM1[0, :])/3),  len(SUM1[0, :])-1]),  ax10.set_xticklabels(['-1', '0', '1', '2'] )
ax10.set_yticks([0, int(len(SUM1)/2),  len(SUM1)-1]),  ax10.set_yticklabels([str(int(min(Betas))), '0' ,str(int(max(Betas)))] )
ax10.axvline(x=int(len(SUM1[0, :])/3), ls ='dashed', color='black')
ax10.axvline(x=beta_slice_mu_index1, ls ='dotted', color=beta_slice_color1)
ax10.axvline(x=beta_slice_mu_index2, ls ='dotted', color=beta_slice_color2)
ax10.axvline(x=beta_slice_mu_index3, ls ='dotted', color=beta_slice_color3)
ax10.axvline(x=beta_slice_mu_index4, ls ='dotted', color=beta_slice_color4)
ax10.axhline(y=mu_slice_beta_index1, ls ='dotted', color=mu_slice_color1)
ax10.axhline(y=int(len(SUM1)/2), ls='dashed', color='black')
ax10.set_xlabel(r'$\mu$')
ax10.set_ylabel(r'$\beta$')
fig.colorbar(HM, ax=ax10, label='Detection index', ticks=[0.2, 0.4, 0.6, 0.8, 1.0])

ax1.text(0.7, 1.2, r'$\tilde{\chi}(\Omega_0)$',  transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax2.text(0.7, 1.2, r'$\tilde{TE}_{FM}$', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax3.text(0.6, 1.2, r'$\tilde{Q}$', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax4.text(0.6, 1.2, r'$\tilde{1/\tau}$', transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax5.text(0.6, 1.2, r'$\tilde{\gamma}$', transform=ax5.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax1.axvline(x=7, ls='dotted', color='black')
ax1.axhline(y=int(len(Betas)/2), ls='dotted', color='black')
ax2.axvline(x=7, ls='dotted', color='black')
ax2.axhline(y=int(len(Betas)/2), ls='dotted', color='black')
ax3.axvline(x=7, ls='dotted', color='black')
ax3.axhline(y=int(len(Betas)/2), ls='dotted', color='black')
ax4.axvline(x=7, ls='dotted', color='black')
ax4.axhline(y=int(len(Betas)/2), ls='dotted', color='black')
ax5.axvline(x=7, ls='dotted', color='black')
ax5.axhline(y=int(len(Betas)/2), ls='dotted', color='black')
ax1.set_xticks([]), ax1.set_yticks([])
ax2.set_xticks([]), ax2.set_yticks([])
ax3.set_xticks([]), ax3.set_yticks([])
ax4.set_xticks([]), ax4.set_yticks([])
ax5.set_xticks([]), ax5.set_yticks([])
ax1.axvline(x=mu_lyap_index1, ls ='dotted', color='yellow')
ax2.axvline(x=mu_lyap_index1, ls ='dotted', color='yellow')
ax3.axvline(x=mu_lyap_index1, ls ='dotted', color='yellow')
ax4.axvline(x=mu_lyap_index1, ls ='dotted', color='yellow')
ax5.axvline(x=mu_lyap_index1, ls ='dotted', color='yellow')
ax1.axvline(x=mu_lyap_index2, ls ='dotted', color='yellow')
ax2.axvline(x=mu_lyap_index2, ls ='dotted', color='yellow')
ax3.axvline(x=mu_lyap_index2, ls ='dotted', color='yellow')
ax4.axvline(x=mu_lyap_index2, ls ='dotted', color='yellow')
ax5.axvline(x=mu_lyap_index2, ls ='dotted', color='yellow')

ax6.plot(Betas_big, beta_slice1, "-", color=beta_slice_color1)
ax6.plot(Betas_big, beta_slice2, "-", color=beta_slice_color2)
ax6.plot(Betas_big, beta_slice3, "-", color=beta_slice_color3)
ax6.plot(Betas_big, beta_slice4, "-", color=beta_slice_color4)
ax6.set_xlabel(r'$\beta$')
ax6.set_ylim(0.4, 1.05)
ax6.set_xlim(-10, 10)
ax7.plot(Mus_big, mu_slice1, "-", color=mu_slice_color1)
ax7.set_xlabel(r'$\mu$')
ax7.set_ylim(0, 1.05)
ax7.set_xticks([-1, 0, 1, 2])
ax7.set_xlim(-1, 2)

ax1.set_xlabel(r'$\mu$')
ax1.set_ylabel(r'$\beta$')
ax2.set_xlabel(r'$\mu$')
ax2.set_ylabel(r'$\beta$')
ax3.set_xlabel(r'$\mu$')
ax3.set_ylabel(r'$\beta$')
ax4.set_xlabel(r'$\mu$')
ax4.set_ylabel(r'$\beta$')
ax5.set_xlabel(r'$\mu$')
ax5.set_ylabel(r'$\beta$')



#big slice from large beta range. Used mu_index=8 about mu=0.14
Lyap_slice1_big = np.array([-0.21805629, -0.12432288,  0.08756548,  0.3202077 ,  0.53525342,
        0.72620759,  0.90212624,  1.06524638,  1.20624616,  1.32317061, 1.39259523])
L_Speed_slice1_big = np.array([0.        , 0.05662832, 0.19470505, 0.35052153, 0.48609999,
       0.60793615, 0.73445969, 0.85834823, 0.94944942, 0.9913152 ,  1.        ])
Lyap_slice1_big = Lyap_slice1_big[:5]
L_Speed_slice1_big = L_Speed_slice1_big[:5]
L_Speed_slice1_big /= max(L_Speed_slice1_big)



ax8.plot(Lyap_slice1, L_Chi_slice1,   "D-", color='black',      markersize=4, label=r'$\tilde{\chi}(\Omega_0)$')
ax8.plot(Lyap_slice1, L_TE_fm_slice1, "^-", color='red',        markersize=4, label=r'$\tilde{TE}_{FM}$')
ax8.plot(Lyap_slice1, L_Q_slice1,     "o-", color='dodgerblue', markersize=4, label=r'$\tilde{Q}$')
#ax8.plot(Lyap_slice1, L_Speed_slice1, "x-", color='green',      markersize=4, label=r'$\tilde{1/\tau}$')
ax8.plot(Lyap_slice1_big, L_Speed_slice1_big, "x-", color='green',      markersize=4, label=r'$\tilde{1/\tau}$')
ax8.plot(Lyap_slice1, L_Gamma_slice1, "s-", color='orange',     markersize=4, label=r'$\tilde{\gamma}$')


ax9.plot(Lyap_slice2, L_Chi_slice2,   "D-", color='black',      markersize=4, label=r'$\tilde{\chi}(\Omega_0)$')
ax9.plot(Lyap_slice2, L_TE_fm_slice2, "^-", color='red',        markersize=4, label=r'$\tilde{TE}_{FM}$')
ax9.plot(Lyap_slice2, L_Q_slice2,     "o-", color='dodgerblue', markersize=4, label=r'$\tilde{Q}$')
ax9.plot(Lyap_slice2, L_Speed_slice2, "x-", color='green',      markersize=4, label=r'$\tilde{1/\tau}$')
ax9.plot(Lyap_slice2, L_Gamma_slice2, "s-", color='orange',     markersize=4, label=r'$\tilde{\gamma}$')
ax8.set_xlabel('Lyapunov exponent')
#ax8.set_ylabel(r'$\tilde{\chi}$, $\tilde{TE}_{FM}$, $\tilde{Q}$, $\tilde{1/\tau}$')
ax9.set_xlabel('Lyapunov exponent')
#ax9.set_ylabel(r'$\tilde{\chi}$, $\tilde{TE}_{FM}$, $\tilde{Q}$, $\tilde{1/\tau}$')
ax8.legend(bbox_to_anchor=(1.07, 0.3), loc="upper left", borderaxespad=0)

ax8.text(-0.1, 1.25, "A", transform=ax8.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax9.text(-0.1, 1.25, "B", transform=ax9.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax10.text(-0.07, 1.09, "C", transform=ax10.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\Fig6.jpeg', dpi=300)






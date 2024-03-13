import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage   #for conour lines

LYAP              = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig7_LYAP_beta25_mu2_0p5cycle_300ic.npy')
LYAP_small        = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig1_LYAP_beta10_mu2_0p5cycle_300ic.npy')
TE_am, TE_am_rev  = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig2_TE_AM_D=0p001_beta25_mu2_N=1M.npy' )
TE_fm, TE_fm_rev  = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig2_TE_FM_D=0p001_beta25_mu2_N=1M.npy' )
Q, BW             = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig7_Q_BW_beta25_mu2_60IC.npy' , allow_pickle="True")
Q, BW_small       = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig4_heatmaps_beta10_mu2_60IC.npy' , allow_pickle="True")
TAU               = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig7_Tau_off_heatmap_beta25_mu2_64ic_5trials.npy' )

LYAP = gaussian_filter(LYAP, 1)  #sigma
LYAP_small = gaussian_filter(LYAP_small, 1)  #sigma
TE_am = gaussian_filter(TE_am, 1)
TE_fm = gaussian_filter(TE_fm, 1)
BW = gaussian_filter(BW, 1)
BW_small = gaussian_filter(BW_small, 1)
TAU = gaussian_filter(TAU, 2)
SPEED = 1/TAU



#Creating Lyaponov slices at constant MU for all 4 measures
length = 11
mu_lyap_index1 = 10   #mu11 about 0.57   mu10 about 0.43
Lyap_slice1      = LYAP[:, mu_lyap_index1].copy()
Lyap_small_slice1 = LYAP_small[:, mu_lyap_index1].copy()
L_TE_am_slice1   = TE_am[:, mu_lyap_index1].copy()
L_BW_slice1      =  BW[:, mu_lyap_index1].copy()
L_BW_small_slice1 =  BW_small[:, mu_lyap_index1].copy()
L_TE_fm_slice1 =  TE_fm[:, mu_lyap_index1].copy()
L_Speed_slice1 =  SPEED[:, mu_lyap_index1].copy()
Lyap_slice1      = (     Lyap_slice1[::-1][length-1:] + Lyap_slice1[-length:])/2    #colapsing both sides onto one
Lyap_small_slice1= (     Lyap_small_slice1[::-1][length-1:] + Lyap_small_slice1[-length:])/2
L_TE_am_slice1   = (  L_TE_am_slice1[::-1][length-1:] + L_TE_am_slice1[-length:])/2
L_BW_slice1      = (     L_BW_slice1[::-1][length-1:] + L_BW_slice1[-length:])/2
L_BW_small_slice1= (     L_BW_small_slice1[::-1][length-1:] + L_BW_small_slice1[-length:])/2
L_TE_fm_slice1   = (  L_TE_fm_slice1[::-1][length-1:] + L_TE_fm_slice1[-length:])/2
L_Speed_slice1   = (  L_Speed_slice1[::-1][length-1:] + L_Speed_slice1[-length:])/2
L_TE_am_slice1 -= min(L_TE_am_slice1)
L_BW_small_slice1-= min(L_BW_slice1)
L_BW_slice1    -= min(L_BW_slice1)
L_TE_fm_slice1 -= min(L_TE_fm_slice1)
L_Speed_slice1 -= min(L_Speed_slice1)
L_TE_am_slice1  /= max(L_TE_am_slice1)
L_BW_slice1     /= max(L_BW_small_slice1)
L_BW_small_slice1  /= max(L_BW_small_slice1)
L_TE_fm_slice1  /= max(L_TE_fm_slice1)
L_Speed_slice1  /= max(L_Speed_slice1)
mu_lyap_index2 = 18  #mu18 about 1.57
Lyap_slice2     =  LYAP[:, mu_lyap_index2].copy()
Lyap_small_slice2 =  LYAP_small[:, mu_lyap_index2].copy()
L_TE_am_slice2  = TE_am[:, mu_lyap_index2].copy()
L_BW_slice2     =    BW[:, mu_lyap_index2].copy()
L_BW_small_slice2 =    BW_small[:, mu_lyap_index2].copy()
L_TE_fm_slice2 =  TE_fm[:, mu_lyap_index2].copy()
L_Speed_slice2 =  SPEED[:, mu_lyap_index2].copy()
Lyap_slice2     = (   Lyap_slice2[::-1][length-1:] + Lyap_slice2[-length:])/2    #colapsing both sides onto one
Lyap_small_slice2= (   Lyap_small_slice2[::-1][length-1:] + Lyap_small_slice2[-length:])/2
L_TE_am_slice2  = (L_TE_am_slice2[::-1][length-1:] + L_TE_am_slice2[-length:])/2
L_BW_slice2     = (   L_BW_slice2[::-1][length-1:] + L_BW_slice2[-length:])/2
L_BW_small_slice2     = (   L_BW_small_slice2[::-1][length-1:] + L_BW_small_slice2[-length:])/2
L_TE_fm_slice2  = (L_TE_fm_slice2[::-1][length-1:] + L_TE_fm_slice2[-length:])/2
L_Speed_slice2  = (L_Speed_slice2[::-1][length-1:] + L_Speed_slice2[-length:])/2
L_TE_am_slice2  -= min(L_TE_am_slice2)
L_BW_small_slice2 -= min(L_BW_slice2)
L_BW_slice2     -= min(L_BW_slice2)
L_TE_fm_slice2  -= min(L_TE_fm_slice2)
L_Speed_slice2  -= min(L_Speed_slice2)
L_TE_am_slice2  /= max(L_TE_am_slice2)
L_BW_slice2     /= max(L_BW_small_slice2)
L_BW_small_slice2     /= max(L_BW_small_slice2)
L_TE_fm_slice2  /= max(L_TE_fm_slice2)
L_Speed_slice2  /= max(L_Speed_slice2)


TE_am     -= TE_am.min()
TE_fm     -= TE_fm.min()
BW        -= BW.min()
BW_small  -= BW_small.min()
SPEED     -= SPEED.min()
quantile = 0.9
TE_am_thresh   = np.quantile(TE_am,   quantile)
TE_fm_thresh   = np.quantile(TE_fm, quantile)
BW_thresh      = np.quantile(BW,     quantile)
BW_small_thresh= np.quantile(BW_small,  quantile)
Speed_thresh   = np.quantile(SPEED, quantile)
TE_am   /= TE_am_thresh
TE_fm   /= TE_fm_thresh
BW      /= BW_thresh
BW_small /= BW_small_thresh
SPEED   /= Speed_thresh
#rectifying at 1
TE_am[TE_am > 1] = 1
TE_fm[TE_fm > 1] = 1
BW[BW > 1]       = 1
BW_small[BW_small > 1]       = 1
SPEED[SPEED > 1] = 1

Mus   = np.linspace(-1, 2, 22)
Betas = np.linspace(-25, 25, 21)
color_map = 'jet'  #jet bwr
fig = plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.93, wspace=3, hspace=0.5)
ax1  = plt.subplot2grid((3, 12), (0,0), colspan=3, rowspan=1)
ax2  = plt.subplot2grid((3, 12), (0,3), colspan=3, rowspan=1)
ax3  = plt.subplot2grid((3, 12), (0,6), colspan=3, rowspan=1)
ax4  = plt.subplot2grid((3, 12), (0,9), colspan=3, rowspan=1)
ax5  = plt.subplot2grid((3, 12), (1,4), colspan=5, rowspan=2)
ax6  = plt.subplot2grid((3, 12), (1,9), colspan=3, rowspan=1)
ax7  = plt.subplot2grid((3, 12), (2,9), colspan=3, rowspan=1)
ax8  = plt.subplot2grid((3, 12), (1,0), colspan=3, rowspan=1)
ax9  = plt.subplot2grid((3, 12), (2,0), colspan=3, rowspan=1)

ax1.imshow(TE_am, cmap=color_map, interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=1)
ax2.imshow(TE_fm, cmap=color_map, interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=1)
ax3.imshow(BW_small,    cmap=color_map, interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=1)
ax4.imshow(SPEED, cmap=color_map, interpolation='bilinear', origin='lower', aspect='auto', vmin=0, vmax=1)

interpolation_factor = 3
beta_slice_color1, beta_slice_color2, beta_slice_color3, beta_slice_color4 = 'mediumblue', 'blue', 'deepskyblue', 'cyan'
mu_slice_color1 = 'magenta'
SUM1 = 1*TE_am  +  1*TE_fm  +  1*BW  +  1*SPEED
SUM1 = gaussian_filter(SUM1, 0.1)
SUM1 = scipy.ndimage.zoom(SUM1, interpolation_factor)  #3 because we interpolated
Mus_big, Betas_big = np.linspace(min(Mus), max(Mus), interpolation_factor*len(Mus)), np.linspace(min(Betas), max(Betas), interpolation_factor*len(Betas))
SUM1 = gaussian_filter(SUM1, 3)
SUM1 /= SUM1.max()  #normalizing
beta_slice_mu_index1 = 10 * interpolation_factor
beta_slice_mu_index2 = 13 * interpolation_factor
beta_slice_mu_index3 = 16 * interpolation_factor
beta_slice_mu_index4 = 21 * interpolation_factor
mu_slice_beta_index1 = 14 * interpolation_factor
beta_slice1 = SUM1[:, beta_slice_mu_index1]
beta_slice2 = SUM1[:, beta_slice_mu_index2]
beta_slice3 = SUM1[:, beta_slice_mu_index3]
beta_slice4 = SUM1[:, beta_slice_mu_index4]
mu_slice1 = SUM1[mu_slice_beta_index1, :]

HM5 = ax5.contourf(SUM1, cmap=color_map, levels=np.linspace(SUM1.min(), SUM1.max(), 80) )   #80 contours good
ax5.set_xticks([0, int(len(SUM1[0, :])/3), int(2*len(SUM1[0, :])/3),  len(SUM1[0, :])-1]),  ax5.set_xticklabels(['-1', '0', '1', '2'] )
ax5.set_yticks([0, int(len(SUM1)/2),  len(SUM1)-1]),  ax5.set_yticklabels([str(int(min(Betas))), '0' ,str(int(max(Betas)))] )
ax5.axvline(x=int(len(SUM1[0, :])/3), ls ='dashed', color='black')
ax5.axvline(x=beta_slice_mu_index1, ls ='dotted', color=beta_slice_color1)
ax5.axvline(x=beta_slice_mu_index2, ls ='dotted', color=beta_slice_color2)
ax5.axvline(x=beta_slice_mu_index3, ls ='dotted', color=beta_slice_color3)
ax5.axvline(x=beta_slice_mu_index4, ls ='dotted', color=beta_slice_color4)
ax5.axhline(y=mu_slice_beta_index1, ls ='dotted', color=mu_slice_color1)
ax5.axhline(y=int(len(SUM1)/2), ls='dashed', color='black')
ax5.set_xlabel(r'$\mu$')
ax5.set_ylabel(r'$\beta$')
fig.colorbar(HM5, ax=ax5, label='Detection index', ticks=[0.2, 0.4, 0.6, 0.8, 1.0])

ax1.text(0.7, 1.2, r'$\tilde{TE}_{AM}$',  transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax2.text(0.7, 1.2, r'$\tilde{TE}_{FM}$', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax3.text(0.6, 1.2, r'$\tilde{BW}$',    transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax4.text(0.6, 1.2, r'$\tilde{1/\tau}$', transform=ax4.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
ax1.axvline(x=7, ls='dotted', color='black')
ax1.axhline(y=int(len(Betas)/2), ls='dotted', color='black')
ax2.axvline(x=7, ls='dotted', color='black')
ax2.axhline(y=int(len(Betas)/2), ls='dotted', color='black')
ax3.axvline(x=7, ls='dotted', color='black')
ax3.axhline(y=int(len(Betas)/2), ls='dotted', color='black')
ax4.axvline(x=7, ls='dotted', color='black')
ax4.axhline(y=int(len(Betas)/2), ls='dotted', color='black')
ax1.set_xticks([]), ax1.set_yticks([])
ax2.set_xticks([]), ax2.set_yticks([])
ax3.set_xticks([]), ax3.set_yticks([])
ax4.set_xticks([]), ax4.set_yticks([])

ax1.axvline(x=mu_lyap_index1, ls ='dotted', color='yellow')
ax2.axvline(x=mu_lyap_index1, ls ='dotted', color='yellow')
ax3.axvline(x=mu_lyap_index1, ls ='dotted', color='yellow')
ax4.axvline(x=mu_lyap_index1, ls ='dotted', color='yellow')
ax1.axvline(x=mu_lyap_index2, ls ='dotted', color='yellow')
ax2.axvline(x=mu_lyap_index2, ls ='dotted', color='yellow')
ax3.axvline(x=mu_lyap_index2, ls ='dotted', color='yellow')
ax4.axvline(x=mu_lyap_index2, ls ='dotted', color='yellow')

ax6.plot(Betas_big, beta_slice1, "-", color=beta_slice_color1)
ax6.plot(Betas_big, beta_slice2, "-", color=beta_slice_color2)
ax6.plot(Betas_big, beta_slice3, "-", color=beta_slice_color3)
ax6.plot(Betas_big, beta_slice4, "-", color=beta_slice_color4)
ax6.set_xlabel(r'$\beta$')
ax6.set_ylim(0, 1.05)
ax6.set_xlim(-25, 25)
ax6.set_xticks([-25, -10, 0, 10, 25])
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

#ax8.plot(Lyap_slice1[3:], L_BW_slice1[3:],    "o-", color='dodgerblue', markersize=4)
#ax8.plot(Lyap_small_slice1[:-4], L_BW_small_slice1[:-4],    "o-", color='dodgerblue', markersize=4)
ax8.plot(Lyap_slice1, L_TE_am_slice1, "d-", color='darkviolet', markersize=4, label=r'$\tilde{TE}_{AM}$')
ax8.plot(Lyap_slice1, L_TE_fm_slice1, "^-", color='red', markersize=4, label=r'$\tilde{TE}_{FM}$')
ax8.plot(np.concatenate([Lyap_small_slice1[:-4], Lyap_slice1[3:]]), np.concatenate( [L_BW_small_slice1[:-4], L_BW_slice1[3:]]),   "+-", color='blue', markersize=4, label=r'$\tilde{BW}$')
ax8.plot(Lyap_slice1, L_Speed_slice1, "x-", color='green', markersize=4, label=r'$\tilde{1/\tau}$')
#ax9.plot(Lyap_slice2, L_BW_slice2,    "o-", color='dodgerblue', markersize=4)
#ax9.plot(Lyap_small_slice2, L_BW_small_slice2,    "o-", color='dodgerblue', markersize=4)
ax9.plot(Lyap_slice2, L_TE_am_slice2, "d-", color='darkviolet', markersize=4, label=r'$\tilde{TE}_{AM}$')
ax9.plot(Lyap_slice2, L_TE_fm_slice2, "^-", color='red', markersize=4, label=r'$\tilde{TE}_{FM}$')
ax9.plot(np.concatenate([Lyap_small_slice2[:-1], Lyap_slice2[5:]]), np.concatenate( [L_BW_small_slice2[:-1], L_BW_slice2[5:]]),   "+-", color='blue', markersize=4, label=r'$\tilde{BW}$')
ax9.plot(Lyap_slice2, L_Speed_slice2, "x-", color='green', markersize=4, label=r'$\tilde{1/\tau}$')
ax8.set_xlabel('Lyapunov exponent')
#ax8.set_ylabel(r'$\tilde{TE}_{FM}$, $\tilde{TE}_{FM}$, $\tilde{BW}$, $\tilde{1/\tau}$')
ax9.set_xlabel('Lyapunov exponent')
#ax9.set_ylabel(r'$\tilde{TE}_{FM}$, $\tilde{TE}_{FM}$, $\tilde{BW}$, $\tilde{1/\tau}$')
ax8.legend(bbox_to_anchor=(1.01, 0.92), loc="upper left", borderaxespad=0)


ax8.text(-0.1, 1.25, "A", transform=ax8.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax9.text(-0.1, 1.25, "B", transform=ax9.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax5.text(-0.07, 1.09, "C", transform=ax5.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\Fig7.jpeg', dpi=300)






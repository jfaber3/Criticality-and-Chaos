import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage   #for conour lines


TE_am, TE_am_rev  = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig2_TE_AM_D=0p001_beta25_mu2_N=1M.npy' )
TE_fm, TE_fm_rev  = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig2_TE_FM_D=0p001_beta25_mu2_N=1M.npy' )
Q, BW             = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig7_Q_BW_beta25_mu2_60IC.npy' , allow_pickle="True")
TAU               = np.load( r'C:\Users\Justin\Desktop\Criticality\Figures_v2\fig7_Tau_off_heatmap_beta25_mu2_64ic_5trials.npy' )


def find_everything(quantile, weights, TE_fm, TE_am, BW, TAU):
	TE_fm = gaussian_filter(TE_fm, 1)
	TE_am = gaussian_filter(TE_am, 1)
	BW = gaussian_filter(BW, 1)
	TAU = gaussian_filter(TAU, 2)
	SPEED = 1/TAU
	TE_fm  -= TE_fm.min()
	TE_am -= TE_am.min()
	BW   -= BW.min()
	SPEED -= SPEED.min()
	TE_fm_thresh = np.quantile(TE_fm, quantile)
	TE_am_thresh = np.quantile(TE_am, quantile)
	BW_thresh     = np.quantile(BW,   quantile)
	Speed_thresh = np.quantile(SPEED, quantile)
	TE_fm /= TE_fm_thresh
	TE_am /= TE_am_thresh
	BW     /= BW_thresh
	SPEED /= Speed_thresh
	#rectifying at 1
	TE_fm[TE_fm > 1] = 1
	TE_am[TE_am > 1] = 1
	BW[BW > 1] = 1
	SPEED[SPEED > 1] = 1
	interpolation_factor = 3
	SUM1 = weights[0]*TE_fm  +  weights[1]*TE_am  +  weights[2]*BW  +  weights[3]*SPEED
	SUM1 = gaussian_filter(SUM1, 0.1)
	SUM1 = scipy.ndimage.zoom(SUM1, interpolation_factor)  #3 because we interpolated
	Mus_big, Betas_big = np.linspace(min(Mus), max(Mus), interpolation_factor*len(Mus)), np.linspace(min(Betas), max(Betas), interpolation_factor*len(Betas))
	SUM1 = gaussian_filter(SUM1, 1)
	SUM1 /= SUM1.max()  #normalizing
	return SUM1, Mus_big, Betas_big


Mus   = np.linspace(-1, 2, 22)
Betas = np.linspace(-25, 25, 21)
color_map = 'jet'  #jet bwr
fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, wspace=3, hspace=1)
ax1  = plt.subplot2grid((5, 15), (0,0), colspan=3, rowspan=1)
ax2  = plt.subplot2grid((5, 15), (0,3), colspan=3, rowspan=1)
ax3  = plt.subplot2grid((5, 15), (0,6), colspan=3, rowspan=1)
ax4  = plt.subplot2grid((5, 15), (0,9), colspan=3, rowspan=1)

ax6  = plt.subplot2grid((5, 15), (1,0), colspan=7, rowspan=2)
ax7  = plt.subplot2grid((5, 15), (1,8), colspan=7, rowspan=2)
ax8  = plt.subplot2grid((5, 15), (3,0), colspan=7, rowspan=2)
ax9  = plt.subplot2grid((5, 15), (3,8), colspan=7, rowspan=2)



quantile_original = 0.9
weights_original = [1, 1, 1, 1, 1]

weights1 = [3, 3, 1, 1]
weights2 = [1, 1, 3, 3]
quantile1 = 0.7
quantile2 = 0.5


SUM1, Mus_big, Betas_big = find_everything(quantile_original, weights1, TE_fm, TE_am, BW, TAU)
HM6 = ax6.contourf(SUM1, cmap=color_map, levels=np.linspace(SUM1.min(), SUM1.max(), 200) )   #80 contours good
ax6.set_xticks([0, int(len(SUM1[0, :])/3), int(2*len(SUM1[0, :])/3),  len(SUM1[0, :])-1]),  ax6.set_xticklabels(['-1', '0', '1', '2'] )
ax6.set_yticks([0, int(len(SUM1)/2),  len(SUM1)-1]),  ax6.set_yticklabels([str(int(min(Betas))), '0' ,str(int(max(Betas)))] )
ax6.axvline(x=int(len(SUM1[0, :])/3), ls ='dashed', color='black')
ax6.axhline(y=int(len(SUM1)/2), ls='dashed', color='black')
ax6.set_xlabel(r'$\mu$')
ax6.set_ylabel(r'$\beta$')
fig.colorbar(HM6, ax=ax6, label='Detection index', ticks=[0.2, 0.4, 0.6, 0.8, 1.0])

SUM1, Mus_big, Betas_big = find_everything(quantile_original, weights2, TE_fm, TE_am, BW, TAU)
HM7 = ax7.contourf(SUM1, cmap=color_map, levels=np.linspace(SUM1.min(), SUM1.max(), 200) )   #80 contours good
ax7.set_xticks([0, int(len(SUM1[0, :])/3), int(2*len(SUM1[0, :])/3),  len(SUM1[0, :])-1]),  ax7.set_xticklabels(['-1', '0', '1', '2'] )
ax7.set_yticks([0, int(len(SUM1)/2),  len(SUM1)-1]),  ax7.set_yticklabels([str(int(min(Betas))), '0' ,str(int(max(Betas)))] )
ax7.axvline(x=int(len(SUM1[0, :])/3), ls ='dashed', color='black')
ax7.axhline(y=int(len(SUM1)/2), ls='dashed', color='black')
ax7.set_xlabel(r'$\mu$')
ax7.set_ylabel(r'$\beta$')
fig.colorbar(HM7, ax=ax7, label='Detection index', ticks=[0.2, 0.4, 0.6, 0.8, 1.0])

SUM1, Mus_big, Betas_big = find_everything(quantile1, weights_original, TE_fm, TE_am, BW, TAU)
HM8 = ax8.contourf(SUM1, cmap=color_map, levels=np.linspace(SUM1.min(), SUM1.max(), 200) )   #80 contours good
ax8.set_xticks([0, int(len(SUM1[0, :])/3), int(2*len(SUM1[0, :])/3),  len(SUM1[0, :])-1]),  ax8.set_xticklabels(['-1', '0', '1', '2'] )
ax8.set_yticks([0, int(len(SUM1)/2),  len(SUM1)-1]),  ax8.set_yticklabels([str(int(min(Betas))), '0' ,str(int(max(Betas)))] )
ax8.axvline(x=int(len(SUM1[0, :])/3), ls ='dashed', color='black')
ax8.axhline(y=int(len(SUM1)/2), ls='dashed', color='black')
ax8.set_xlabel(r'$\mu$')
ax8.set_ylabel(r'$\beta$')
fig.colorbar(HM8, ax=ax8, label='Detection index', ticks=[0.2, 0.4, 0.6, 0.8, 1.0])

SUM1, Mus_big, Betas_big = find_everything(quantile2, weights_original, TE_fm, TE_am, BW, TAU)
HM9 = ax9.contourf(SUM1, cmap=color_map, levels=np.linspace(SUM1.min(), SUM1.max(), 200) )   #80 contours good
ax9.set_xticks([0, int(len(SUM1[0, :])/3), int(2*len(SUM1[0, :])/3),  len(SUM1[0, :])-1]),  ax9.set_xticklabels(['-1', '0', '1', '2'] )
ax9.set_yticks([0, int(len(SUM1)/2),  len(SUM1)-1]),  ax9.set_yticklabels([str(int(min(Betas))), '0' ,str(int(max(Betas)))] )
ax9.axvline(x=int(len(SUM1[0, :])/3), ls ='dashed', color='black')
ax9.axhline(y=int(len(SUM1)/2), ls='dashed', color='black')
ax9.set_xlabel(r'$\mu$')
ax9.set_ylabel(r'$\beta$')
fig.colorbar(HM9, ax=ax9, label='Detection index', ticks=[0.2, 0.4, 0.6, 0.8, 1.0])



hist_bins = 30
ax1.hist(TE_fm.flatten(),     bins=hist_bins, color='black')
ax2.hist(TE_am.flatten(),   bins=hist_bins, color='black')
ax3.hist(BW.flatten(),       bins=hist_bins, color='black')
ax4.hist((1/TAU).flatten(), bins=hist_bins, color='black')


TE_fm_sat1, TE_fm_sat2, TE_fm_sat3    = np.quantile(TE_fm,  quantile_original), np.quantile(TE_fm,  quantile1), np.quantile(TE_fm,  quantile2)
TE_am_sat1, TE_am_sat2, TE_am_sat3    = np.quantile(TE_am,  quantile_original), np.quantile(TE_am,  quantile1), np.quantile(TE_am,  quantile2)
BW_sat1, BW_sat2, BW_sat3             = np.quantile(BW,  quantile_original), np.quantile(BW,  quantile1), np.quantile(BW,  quantile2)
Speed_sat1, Speed_sat2, Speed_sat3       = np.quantile((1/TAU),  quantile_original), np.quantile((1/TAU),  quantile1), np.quantile((1/TAU),  quantile2)



BW_sat2 = 0.02
BW_sat3 = 0.02  #just above zero



ax1.axvline(x=TE_fm_sat1, ls='dotted', color='dodgerblue')
ax1.axvline(x=TE_fm_sat2, ls='dotted', color='green')
ax1.axvline(x=TE_fm_sat3, ls='dotted', color='red')
ax2.axvline(x=TE_am_sat1, ls='dotted', color='dodgerblue')
ax2.axvline(x=TE_am_sat2, ls='dotted', color='green')
ax2.axvline(x=TE_am_sat3, ls='dotted', color='red')
ax3.axvline(x=BW_sat1, ls='dotted', color='dodgerblue')
ax3.axvline(x=BW_sat2, ls='dotted', color='green')
ax3.axvline(x=BW_sat3, ls='dotted', color='red')
ax4.axvline(x=Speed_sat1, ls='dotted', color='dodgerblue')
ax4.axvline(x=Speed_sat2, ls='dotted', color='green')
ax4.axvline(x=Speed_sat3, ls='dotted', color='red')


ax1.set_ylabel('Count')
ax1.set_xlabel(r'$TE_{FM}$')
ax2.set_xlabel(r'$TE_{AM}$')
ax3.set_xlabel(r'$BW$')
ax4.set_xlabel(r'$1/\tau$')

ax1.text(-0.15, 1.2, "A", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax6.text(-0.1, 1.1, "B", transform=ax6.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax7.text(-0.1, 1.1, "C", transform=ax7.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax8.text(-0.1, 1.1, "D", transform=ax8.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax9.text(-0.1, 1.1, "E", transform=ax9.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\FigS4.jpeg', dpi=300)






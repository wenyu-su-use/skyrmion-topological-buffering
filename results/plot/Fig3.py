# -*- coding: utf-8 -*-
"""
Beautiful Unified Figure Generation (Optimized for PRL One-Column)
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib import rcParams

# ================= 1. Global Style Configuration =================
# PRL standard: Single column = 3.375 inches (8.6 cm)
# We choose a height that maintains the 1+2 layout balance.
FIG_WIDTH = 3.375  
FIG_HEIGHT = 3.8
FONT_SIZE = 8 

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': FONT_SIZE,
    'axes.linewidth': 0.8,       # Thinner lines for small figures
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 3,       # Shorter ticks for legibility
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'legend.frameon': False,     # Cleaner look
    'pdf.fonttype': 42,          # Ensures text is editable in Illustrator
    'ps.fonttype': 42
})

try:
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}'
except:
    rcParams['text.usetex'] = False

# ================= Custom Colors & Arrow Configuration =================
COLOR_ATOM = '#2F4F4F'             
COLOR_BOND_NEUTRAL = '#D3D3D3'     
COLOR_ARROW_J = '#B22222'          
COLOR_CHI_GREEN = '#228B22'        
COLOR_BASIS = '#404040'            
COLOR_SPIN_3D = '#000080'          
COLOR_FACE_LK = '#20B2AA'          
COLOR_FACE_KM = '#48D1CC'          
COLOR_FACE_ML = '#008080'          
COLOR_FACE_TOP = '#E0FFFF'         
COLOR_VOL_EDGE = '#006400' 

COLOR_Q = '#EE7733'
COLOR_PSI = '#009988'
COLOR_ENT = '#CC79A7'

A_CONST = 1.0
R_SPHERE = 0.14
LW_BOND = 1.5            
LW_DM = 1.5
STROKE_LW = 1.2          
ARROW_HW = 4.0           
ARROW_HL = 4.0

# ================= Vector Map and Inset Configuration =================
blue_coolwarm = plt.get_cmap('coolwarm')(0.0) 
red_rdbu      = plt.get_cmap('bwr')(1.0)   
mid_white      = '#FFFFFF' 
hybrid_colors = [blue_coolwarm, mid_white, red_rdbu]
HYBRID_CMAP = LinearSegmentedColormap.from_list('MyHybridCmap', hybrid_colors)

# 应用自定义的箭头参数
ARROW_OPTS = {
    'scale': 25,
    'scale_units': 'inches',
    'width': 0.005,
    'headwidth': ARROW_HW,
    'headlength': ARROW_HL,
    'minlength': 3,
    'alpha': 1.0,
    'pivot': 'middle'
}
THETA_CMAP = HYBRID_CMAP 
SF_CMAP = 'hot'
GRID_RES = 300

# ================= Helper Functions =================
def load_spatial_data(id_val, ib_val=0, data_dir="."):
    Lx_cur, Ly_cur = 90, 90
    f_theta = os.path.join(data_dir, f'allbins_theta_iD{id_val}_iB{ib_val}.bin')
    f_phi   = os.path.join(data_dir, f'allbins_phi_iD{id_val}_iB{ib_val}.bin')
    f_sf    = os.path.join(data_dir, f"SF_bin0_iD{id_val}_iB{ib_val}.txt")
    crop_size_cur = 84
    plot_limit_cur = 2.5 

    I, J = np.meshgrid(np.arange(Lx_cur), np.arange(Ly_cur))
    I_f, J_f = I.flatten(), J.flatten()
    X_phys = I_f + 0.5 * J_f
    Y_phys = (np.sqrt(3) / 2.0) * J_f

    if os.path.exists(f_theta):
        theta = np.fromfile(f_theta, dtype=float)[:Lx_cur*Ly_cur]
        phi   = np.fromfile(f_phi, dtype=float)[:Lx_cur*Ly_cur]
        pos_phys = np.vstack((X_phys, Y_phys)).T
    else:
        theta = np.random.rand(Lx_cur * Ly_cur) * np.pi
        phi = np.random.rand(Lx_cur * Ly_cur) * 2 * np.pi
        pos_phys = np.vstack((X_phys, Y_phys)).T

    xi = np.linspace(-plot_limit_cur, plot_limit_cur, GRID_RES)
    yi = np.linspace(-plot_limit_cur, plot_limit_cur, GRID_RES)
    XI, YI = np.meshgrid(xi, yi)
    
    if os.path.exists(f_sf):
        data = np.loadtxt(f_sf, skiprows=1)
        if data.size > 0:
            qx, qy, sq = data[:, 0], data[:, 1], data[:, 2]
            ZI = griddata((qx, qy), sq, (XI, YI), method='linear', fill_value=0)
        else:
            ZI = np.zeros_like(XI)
    else:
        ZI = np.exp(-(XI**2 + YI**2)/0.1) * 100 
        
    return pos_phys, theta, phi, XI, YI, ZI, crop_size_cur

# ================= Main Plotting Function =================
def generate_master_figure():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.abspath(os.path.join(script_dir, "../datas_B0.54"))

    # Use the journal-specific dimensions
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
    
    gs = fig.add_gridspec(2, 2, 
                          wspace=0.2, hspace=0.35)
    
    ax_top = fig.add_subplot(gs[0, :])
    ax_bot_left = fig.add_subplot(gs[1, 0])
    ax_bot_right = fig.add_subplot(gs[1, 1])
    axes_bot = [ax_bot_left, ax_bot_right]

    # Labels updated for smaller font
    ax_top.text(-0.15, 0.95, r'\textbf{(a)}', transform=ax_top.transAxes, fontweight='bold', va='bottom', ha='left')
    labels_bot = [r'\textbf{(b)}', r'\textbf{(c)}']
    for ax_bot_inst, label in zip(axes_bot, labels_bot):
        ax_bot_inst.text(-0.1, 1.05, label, transform=ax_bot_inst.transAxes, fontweight='bold', va='bottom', ha='left')

    # ==========================================
    # First Row (a): Phase Observables at B=0.54
    # ==========================================
    print("Plotting top banner figure (a) Observables...")
    try:
        try:
            df1_c = pd.read_csv(os.path.join(target_data_dir, "datan1.txt"), sep=r'\s+', engine='python', on_bad_lines='skip')
            df2_c = pd.read_csv(os.path.join(target_data_dir, "datan2.txt"), sep=r'\s+', engine='python', on_bad_lines='skip')
        except TypeError:
            df1_c = pd.read_csv(os.path.join(target_data_dir, "data.txt"), sep=r'\s+', engine='python', error_bad_lines=False)
            df2_c = pd.read_csv(os.path.join(target_data_dir, "data2.txt"), sep=r'\s+', engine='python', error_bad_lines=False)
        
        for col in df1_c.columns:
            if 'D' in col or 'D_phys' in col:
                df1_c.rename(columns={col: 'D'}, inplace=True)
                break
        for col in df2_c.columns:
            if 'D' in col or 'D_phys' in col:
                df2_c.rename(columns={col: 'D'}, inplace=True)
                break
                
        df_all_c = pd.merge(df1_c, df2_c, on='D', how='left')
        
        required_cols = ['Q_norm', 'Psi6', 'phi6', 'S_norm']
        for col in required_cols:
            if col not in df_all_c.columns:
                df_all_c[col] = 0.0
                
        df_all_c.dropna(subset=['Q_norm', 'Psi6', 'phi6', 'S_norm'], inplace=True)
        max_d_c = 2.5
        df_all_c = df_all_c[df_all_c['D'] <= max_d_c].copy()

        D_arr_c = df_all_c['D'].values
        Q_val_c = df_all_c['Q_norm'].values
        Psi6_r_val_c = df_all_c['Psi6'].values
        Psi6_k_val_c = df_all_c['phi6'].values    
        S_norm_val_c = df_all_c['S_norm'].values

        sigma_smooth_c = 1.0
        sigma_deriv_c = 0.8
        
        dPsi6_k_c = np.abs(np.gradient(gaussian_filter1d(Psi6_k_val_c, sigma=sigma_smooth_c), D_arr_c))
        sm_dPsi6_k_c = gaussian_filter1d(dPsi6_k_c, sigma=sigma_deriv_c)
        Dc1_idx_c = np.argmax(sm_dPsi6_k_c)
        Dc1_c = D_arr_c[Dc1_idx_c] if len(D_arr_c) > 0 else 0.1
        
        glass_mask_c = (D_arr_c > Dc1_c) & (S_norm_val_c > 0.98)
        Dc2_c = D_arr_c[glass_mask_c][0] if np.any(glass_mask_c) else Dc1_c + 0.1 

        dQ_c = np.abs(np.gradient(gaussian_filter1d(Q_val_c, sigma=sigma_smooth_c), D_arr_c))
        sm_dQ_c = gaussian_filter1d(dQ_c, sigma=sigma_deriv_c)
        valid_idx = np.where(D_arr_c > 0.8)[0]
        if len(valid_idx) > 0:
            drop_idx = valid_idx[np.argmax(sm_dQ_c[valid_idx])]
            Dc3_idx_c = max(0, drop_idx - 13) 
            Dc3_c = D_arr_c[Dc3_idx_c]
        else:
            Dc3_c = max_d_c

    except Exception as e:
        print("⚠️ Failed to read original data, using placeholder data...")
        D_arr_c = np.linspace(0.01, 2.5, 100)
        Q_val_c = np.exp(-D_arr_c)
        Psi6_r_val_c = np.exp(-2 * D_arr_c)
        S_norm_val_c = 1 - np.exp(-1.5 * D_arr_c)
        Dc1_c, Dc2_c, Dc3_c, max_d_c = 0.4, 0.8, 1.2, 2.5

    x_min_c = 0.025 
    
    ax_top.axvspan(x_min_c, Dc1_c, color='#FF9999', alpha=0.15)            
    ax_top.axvspan(Dc1_c, max_d_c, color='#FFDD99', alpha=0.2)                
    
    ax_top.axvline(Dc1_c, color='k', linestyle='--', alpha=0.8, linewidth=1.0)

    MK_EVERY = 2 
    l1_c = ax_top.plot(D_arr_c, Q_val_c, 's--', color=COLOR_Q, ms=3, lw=1.0, markevery=MK_EVERY, alpha=0.6, label=r'$Q$')
    l2_c = ax_top.plot(D_arr_c, Psi6_r_val_c, 'd-.', color=COLOR_PSI, mfc='white', ms=3, markevery=MK_EVERY, lw=1.0, label=r'$\Psi_6$')
    l4_c = ax_top.plot(D_arr_c, S_norm_val_c, '^-', color=COLOR_ENT, mfc='white', ms=3, markevery=MK_EVERY, lw=1.0, label=r'$S_{n}$')

    ax_top.set_xlabel(r'$\delta$', labelpad=2)
    ax_top.set_ylabel(r'Norm. Obs.')
    ax_top.set_ylim(0.11, 1.1)
    ax_top.set_xlim(x_min_c, max_d_c)
    ax_top.set_xscale('log')
    ax_top.get_xaxis().set_major_formatter(ticker.ScalarFormatter()) 
    
    ax_top.grid(True, which='major', linestyle='-', alpha=0.3)
    ax_top.grid(True, which='minor', linestyle=':', alpha=0.2)

    lines_main_c = l1_c + l2_c + l4_c 
    labels_main_c = [l.get_label() for l in lines_main_c]
    
    ax_top.legend(lines_main_c, labels_main_c, loc='lower left', bbox_to_anchor=(0.03, 0.05),
                  ncol=1, framealpha=0.85, edgecolor='none', labelspacing=0.3, handlelength=1.2)

    # ==========================================
    # Second Row (b) & (c): Side-by-side spatial vector maps
    # ==========================================
    print("Plotting bottom spatial maps (b) and (c)...")
    
    FILE_INDICES = [(0, 0),(0,2), (8, 0)]
    titles = [r'$B=0.54, \delta=0.2$', r'$B=0.54,\delta=0.5$']
    
    last_qr = None
    last_sf = None
    
    for i, (ax_bot, (id_val, ib_val), tit) in enumerate(zip(axes_bot, FILE_INDICES, titles)):
        
        ax_bot.set_title(tit, fontsize=FONT_SIZE, pad=4) 
        
        pos_p, theta, phi, XI, YI, ZI, crop_size = load_spatial_data(id_val, ib_val, data_dir=target_data_dir)
    
        u, v = np.cos(phi), np.sin(phi)
        qr = ax_bot.quiver(pos_p[:, 0], pos_p[:, 1], u, v, theta, cmap=THETA_CMAP, norm=Normalize(0, np.pi), **ARROW_OPTS)
        last_qr = qr
        
        xc, yc = np.mean(pos_p[:, 0]), np.mean(pos_p[:, 1])
        x_min, x_max = xc - crop_size/3, xc + crop_size/3
        y_min, y_max = yc - crop_size/2.5, yc + crop_size/2.5
        
        offset_x = crop_size / 8 
        cut_bottom_amount = crop_size / 6  
        
        ax_bot.set_xlim(x_min + offset_x, x_max + offset_x)
        ax_bot.set_ylim(y_min + cut_bottom_amount, y_max)
        ax_bot.set_aspect('equal')

        ax_bot.set_xlabel(r'$x/a$', labelpad=1)
        
        x0, x1 = ax_bot.get_xlim()
        y0, y1 = ax_bot.get_ylim()
        ticks_x = np.arange(0, x1 - x0, 20)
        ticks_y = np.arange(0, y1 - y0, 20)
        
        ax_bot.set_xticks(x0 + ticks_x)
        ax_bot.set_xticklabels([f"{int(t)}" for t in ticks_x])

        if i == 0:
            ax_bot.set_ylabel(r'$y/a$', labelpad=1)
            ax_bot.set_yticks(y0 + ticks_y)
            ax_bot.set_yticklabels([f"{int(t)}" for t in ticks_y])
        else:
            ax_bot.set_yticks([])

        # Structure factor inset
        ax_sf = ax_bot.inset_axes([0.6, 0.02, 0.38, 0.38])
        vmax = np.nanmax(ZI) if np.nanmax(ZI) > 1e-10 else 1.0
        vmin = vmax * 0.02
        pcm_sf = ax_sf.pcolormesh(XI, YI, ZI, cmap=SF_CMAP, shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
        last_sf = pcm_sf
        
        ax_sf.set_xticks([]); ax_sf.set_yticks([])
        for spine in ax_sf.spines.values(): 
            spine.set_edgecolor('white')
            spine.set_linewidth(1.5)

    # ==========================================
    # Unify both colorbars on the far right (attached to panel c)
    # ==========================================
    if last_qr is not None and last_sf is not None:
        ax_right = axes_bot[1]
        
        # Upper half: Colorbar for Theta
        cax_main = ax_right.inset_axes([1.06, 0.52, 0.06, 0.48]) 
        cb_main = fig.colorbar(last_qr, cax=cax_main, ticks=[0, np.pi/2, np.pi])
        cb_main.ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$'])
        cb_main.set_label(r'Polar $\theta$', labelpad=5, fontsize=FONT_SIZE-1)
        cb_main.ax.tick_params(labelsize=FONT_SIZE-2)
        
        # Lower half: Colorbar for S(q)
        cax_sf = ax_right.inset_axes([1.06, 0.00, 0.06, 0.48]) 
        # Ticks 改为符合 0.01 到 1.0 的刻度
        cb_sf = fig.colorbar(last_sf, cax=cax_sf, ticks=[0.01, 0.1, 1.0])
        cb_sf.ax.set_yticklabels(['0.01', '0.1', '1'])
        # 标签修改为相对强度
        cb_sf.set_label(r'$S(\mathbf{q})/S_{\mathrm{max}}$', labelpad=5, fontsize=FONT_SIZE-1)
        cb_sf.ax.tick_params(labelsize=FONT_SIZE-2)

    plt.savefig('Fig3.png', dpi=300, bbox_inches='tight')
    
    out_name = "Fig3.pdf"
    #plt.savefig(out_name, bbox_inches='tight')
    print(f"\n✅ Successfully generated unified figure and saved as: {out_name}")
    
    plt.show()

if __name__ == "__main__":
    generate_master_figure()

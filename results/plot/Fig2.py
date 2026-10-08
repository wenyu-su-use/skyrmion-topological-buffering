# -*- coding: utf-8 -*-
"""
Unified Asymmetric Figure Generation (3 Top + 4 Bottom)
Row 1: 
  (a) Raw Phase Diagram (Enlarged)
  (b) Phase Observables (B=1.2) - Expanded to fill middle space
  (c) Autocorrelation (Formerly d)
Row 2: 
  (d)-(g) Spatial Vector Maps & S(q) (Formerly e-h)
"""

import os
import io
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

# ================= 1. Global Master Style Configuration (PRL Standard) =================
FIG_WIDTH_INCHES = 6.77  # 17.2 cm (PRL maximum 2-column width)
FIG_HEIGHT_INCHES = 4.0  # Finely tuned for the new aspect ratios
FONT_SIZE = 9            # True size at 100% scale

mpl.rcParams.update({
    'figure.figsize': (FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES),
    'figure.dpi': 300,
    
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': FONT_SIZE,
    
    'axes.linewidth': 0.8,
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.minor.size': 2.0,
    'ytick.minor.size': 2.0,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    
    'legend.fontsize': FONT_SIZE - 2, 
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight'
})

try:
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}'
except:
    rcParams['text.usetex'] = False

# ================= Exclusive Color Palette =================
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

COLOR_M = '#004488'
COLOR_Q = '#EE7733'
COLOR_PSI = '#009988'        

A_CONST = 1.0
R_SPHERE = 0.14
LW_BOND = 1.5            
LW_DM = 1.5
STROKE_LW = 1.2          
ARROW_HW = 4.0           
ARROW_HL = 4.0

blue_coolwarm = plt.get_cmap('coolwarm')(0.0) 
red_rdbu      = plt.get_cmap('bwr')(1.0)   
mid_white     = '#FFFFFF' 
hybrid_colors = [blue_coolwarm, mid_white, red_rdbu]
HYBRID_CMAP = LinearSegmentedColormap.from_list('MyHybridCmap', hybrid_colors)

# User's finely tuned sharp arrow physics
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
def load_spatial_data(id_val, ib_val=0, data_dir=".", frame_idx=-1):
    """
    修改为支持任意帧抽取：
    frame_idx: 指定读取文件中的第几个 Lx_cur*Ly_cur 块。
               0 为第一帧，1 为第二帧，-1 代表最后一帧。
    """
    Lx_cur, Ly_cur = 90, 90
    f_theta = os.path.join(data_dir, f'allbins_theta_iD{id_val}_iB{ib_val}.bin')
    f_phi   = os.path.join(data_dir, f'allbins_phi_iD{id_val}_iB{ib_val}.bin')
    f_sf    = os.path.join(data_dir, f"SF_allaveraged_iD{id_val}_iB{ib_val}.txt")
    crop_size_cur = 88
    plot_limit_cur = 2.5 

    I, J = np.meshgrid(np.arange(Lx_cur), np.arange(Ly_cur))
    I_f, J_f = I.flatten(), J.flatten()
    X_phys = I_f + 0.5 * J_f
    Y_phys = (np.sqrt(3) / 2.0) * J_f

    if os.path.exists(f_theta):
        # 1. 完整读取文件
        theta_all = np.fromfile(f_theta, dtype=float)
        phi_all   = np.fromfile(f_phi, dtype=float)
        
        # 2. 计算文件的总帧数
        elements_per_frame = Lx_cur * Ly_cur
        total_frames = len(theta_all) // elements_per_frame
        
        # 3. 解析 frame_idx 定位实际帧
        if total_frames > 0:
            if frame_idx < 0:
                # 支持负数索引，例如 -1 为倒数第一帧
                actual_idx = max(0, total_frames + frame_idx)
            else:
                # 防止索引越界
                actual_idx = min(frame_idx, total_frames - 1)
        else:
            actual_idx = 0
            
        # 4. 提取该帧的数据
        start_idx = actual_idx * elements_per_frame
        end_idx = start_idx + elements_per_frame
        
        theta = theta_all[start_idx:end_idx]
        phi   = phi_all[start_idx:end_idx]
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

def log_bin_statistics(time, data, num_bins=200):
    valid_mask = time > 0
    time = time[valid_mask]
    data = data[valid_mask]
    if len(time) == 0: return np.array([]), np.array([]), np.array([])
    t_min, t_max = time.min(), time.max()
    bins = np.geomspace(t_min, t_max, num_bins + 1)
    indices = np.digitize(time, bins)
    t_binned, y_mean, y_std = [], [], []
    for i in range(1, num_bins + 1):
        mask = indices == i
        if np.any(mask):
            current_t = time[mask]
            current_y = data[mask]
            t_binned.append(np.exp(np.mean(np.log(current_t))))
            y_mean.append(np.mean(current_y))
            y_std.append(np.std(current_y))
    return np.array(t_binned), np.array(y_mean), np.array(y_std)

# ================= Unified Plotting Main Function =================
def generate_master_figure():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.abspath(os.path.join(script_dir, "../datas_phase"))

    fig = plt.figure()
    
    # ---------------------------------------------------------
    # THE MASTERCLASS LAYOUT: DECOUPLED ROWS
    # ---------------------------------------------------------
    # Bottom Row: 4 Equal Panels (d, e, f, g)
    gs_bot = fig.add_gridspec(1, 4, left=0.06, right=0.96, bottom=0.08, top=0.42, wspace=0.35)
    
    # Top Row: 3 Custom Columns with dedicated width ratios
    gs_top = fig.add_gridspec(1, 3, left=0.06, right=1.01, bottom=0.55, top=0.9, 
                              width_ratios=[1.1, 1.8, 0.85], wspace=0.48)
    
    # Assign (a) to the first large slot
    ax_a = fig.add_subplot(gs_top[0, 0]) 
    
    # Assign (b) to fully occupy the middle slot natively (No subgridspec needed)
    ax_b = fig.add_subplot(gs_top[0, 1])
    
    # User's manual visual nudge for (b)
    pos = ax_b.get_position() 
    ax_b.set_position([pos.x0 + 0.02, pos.y0, pos.width, pos.height - 0.02])
    
    # Assign (c) to the right slot (Formerly autocorrelation d)
    ax_c = fig.add_subplot(gs_top[0, 2])
    
    # Assign Bottom Row (Formerly efgh, now defg)
    axes_defg = [
        fig.add_subplot(gs_bot[0, 0]),
        fig.add_subplot(gs_bot[0, 1]),
        fig.add_subplot(gs_bot[0, 2]),
        fig.add_subplot(gs_bot[0, 3])
    ]

    # Assign Panel Labels (Sequentially Re-lettered)
    ax_a.text(-0.15, 1.08, '(a)', transform=ax_a.transAxes, fontsize=FONT_SIZE+1, fontweight='bold', va='top', ha='right')
    ax_b.text(-0.08, 1.08, '(b)', transform=ax_b.transAxes, fontsize=FONT_SIZE+1, fontweight='bold', va='top', ha='right')
    ax_c.text(-0.15, 1.08, '(c)', transform=ax_c.transAxes, fontsize=FONT_SIZE+1, fontweight='bold', va='top', ha='right')

    for ax, label in zip(axes_defg, ['(d)', '(e)', '(f)', '(g)']):
        ax.text(-0.12, 1.12, label, transform=ax.transAxes, fontsize=FONT_SIZE+1, fontweight='bold', va='top', ha='right')
        
    # ==========================================
    # Subplot (a): Raw Phase Diagram
    # ==========================================
    print("Plotting (a)...")
    data_bg_file = os.path.join(target_data_dir, "Master_Merged_Data_Complete.txt")
    
    if os.path.exists(data_bg_file):
        data_bg = np.loadtxt(data_bg_file, comments='#')
        B_bg, D_bg, Q_bg = data_bg[:, 0], data_bg[:, 1], np.abs(data_bg[:, 4])

        master_file_raw = os.path.join(target_data_dir, "All_Extracted_Data_Sorted.txt")
        master_file_norm = os.path.join(target_data_dir, "All_Extracted_Data_Sorted_Normalized.txt")
        
        if os.path.exists(master_file_raw) and os.path.exists(master_file_norm):
            data_raw = np.loadtxt(master_file_raw, comments='#')
            data_norm = np.loadtxt(master_file_norm, comments='#')
            master_data = np.vstack((data_raw[data_raw[:, 1] <= 1.0], data_norm[data_norm[:, 1] > 1.0]))
            B_master, D_master = master_data[:, 0], master_data[:, 1]
            Q_master, P6_master = np.abs(master_data[:, 4]), master_data[:, 6]

            B_ent, D_ent, Snorm_ent = [], [], []
            search_pattern = os.path.join(target_data_dir, "KTHNY_Entropy_Statistics_B*.txt")
            for f in glob.glob(search_pattern):
                filename = os.path.basename(f)
                match = re.search(r'B([\d\.]+)\.txt', filename)
                if match:
                    try:
                        data = np.atleast_2d(np.loadtxt(f))
                        for row in data:
                            B_ent.append(float(match.group(1)))
                            D_ent.append(row[0])
                            Snorm_ent.append(row[5])
                    except: continue
            B_ent, D_ent, Snorm_ent = np.array(B_ent), np.array(D_ent), np.array(Snorm_ent)

            min_B, max_B, min_D, max_D = 0.01, 2.0, 0.0, 1.5
            grid_B, grid_D = np.mgrid[min_B:max_B:600j, min_D:max_D:500j] 
            
            grid_Q_bg = griddata((B_bg, D_bg), Q_bg, (grid_B, grid_D), method='nearest')
            grid_P6 = griddata((B_master, D_master), P6_master, (grid_B, grid_D), method='nearest')
            grid_Q_bnd = griddata((B_master, D_master), Q_master, (grid_B, grid_D), method='nearest')
            grid_Snorm = griddata((B_ent, D_ent), Snorm_ent, (grid_B, grid_D), method='nearest')

            mask_brg = (grid_Snorm <= 0.92) & (grid_Q_bnd > 0.9)
            mask_skg = (grid_Snorm > 0.92) & (grid_Q_bnd > 0.95)
            mask_skx = (grid_B >= 0.4) & (grid_P6 > 0.95)
            mask_phase4 = (grid_Snorm <= 0.92) & (grid_Q_bnd < 0.6)

            pcm = ax_a.pcolormesh(grid_B, grid_D, grid_Q_bg, cmap='RdYlBu_r', shading='nearest', vmin=0, vmax=1.0, rasterized=True)
            line_kw = {'colors': 'white', 'linestyles': '--', 'linewidths': 1.0}
            ax_a.contour(grid_B, grid_D, mask_brg.astype(float), levels=[0.5], **line_kw)
            ax_a.contour(grid_B, grid_D, mask_skg.astype(float), levels=[0.5], **line_kw)
            ax_a.contour(grid_B, grid_D, mask_skx.astype(float), levels=[0.5], **line_kw)
            ax_a.contour(grid_B, grid_D, mask_phase4.astype(float), levels=[0.5], **line_kw)

            def add_lbl(mask, txt, ox=0, oy=0):
                if np.any(mask):
                    y, x = np.where(mask)
                    ax_a.text(np.median(grid_B[y,x])+ox, np.median(grid_D[y,x])+oy, txt, 
                              color='white', fontsize=FONT_SIZE, fontweight='bold', ha='center', va='center',
                              path_effects=[pe.withStroke(linewidth=1.8, foreground='#111111')])
            add_lbl(mask_skx, "SkX",oy=0.02); add_lbl(mask_brg, "BrG", oy=0.06)
            add_lbl(mask_skg, "SkG",oy=-0.04,ox=-0.04); add_lbl(mask_phase4, "H", ox=0.15,oy=-0.03)
            add_lbl(mask_skg, "TBP", ox=-0.2,oy=0.58)
            
            cbar = fig.colorbar(pcm, ax=ax_a, fraction=0.046, pad=0.03)
            cbar.ax.set_yticks([0., 0.5, 1.0])
            cbar.ax.tick_params(labelsize=FONT_SIZE)
            cbar.ax.set_ylabel(r'$Q$')
            
            ax_a.set_xlim(0.3, max_B)
            ax_a.set_ylim(0.05, max_D)
            
            # Now it is safe to apply the log scales
            ax_a.set_xscale('log')
            ax_a.set_yscale('log')
            
            # Format the axes cleanly
            ax_a.set_xticks([0.5, 1.0, 2.0])
            ax_a.set_yticks([0.1, 0.5, 1.0]) # Add explicit log ticks for the y-axis
            
            ax_a.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax_a.get_xaxis().set_minor_formatter(ticker.NullFormatter()) 
            ax_a.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
            ax_a.get_yaxis().set_minor_formatter(ticker.NullFormatter())
            ax_a.set_xlabel(r'$B$'); ax_a.set_ylabel(r'$\delta$')
    else:
        print(f"⚠️ Background phase data missing: {data_bg_file}")

    # ==========================================
    # Subplot (b): Phase Observables B=1.2 
    # ==========================================
    print("Plotting (b)...")
    data1_str_b = """0.00    0.997317    0.991526    0.473776\n0.01    0.998108    0.991259    0.473832\n0.02    1.000000    0.988754    0.473801\n0.03    0.998902    0.990193    0.473918\n0.04    0.998108    0.989155    0.474096\n0.05    0.999085    0.987401    0.474221\n0.06    0.998413    0.987076    0.474482\n0.07    0.998108    0.985142    0.474721\n0.08    0.999268    0.981160    0.474871\n0.09    0.998718    0.978418    0.475274\n0.10    0.998413    0.971502    0.475694\n0.11    0.996949    0.964090    0.476213\n0.15    0.995305    0.917617    0.478480\n0.20    0.995854    0.812418    0.482077\n0.25    0.996341    0.727874    0.485210\n0.30    1.000000    0.666212    0.487791\n0.35    0.998388    0.625880    0.490096\n0.40    0.998668    0.585047    0.492186\n0.45    0.998280    0.564553    0.494231\n0.50    0.996674    0.540305    0.496213\n0.55    0.994785    0.514813    0.498957\n0.60    0.992599    0.501528    0.500832\n0.65    0.987454    0.487147    0.502575\n0.70    0.978568    0.467919    0.505834\n0.75    0.972068    0.458111    0.507902\n0.80    0.961500    0.446095    0.511180\n0.85    0.950318    0.437883    0.513465\n0.90    0.935965    0.426497    0.516760\n0.95    0.920015    0.414016    0.519855\n1.00    0.898222    0.407995    0.522180\n1.05    0.883870    0.400237    0.524521\n1.10    0.856373    0.390740    0.529581\n1.15    0.834522    0.383522    0.531049\n1.20    0.810886    0.375653    0.534962\n1.25    0.782252    0.370760    0.537580\n1.30    0.760378    0.363810    0.540523\n1.35    0.732372    0.355498    0.543617\n1.40    0.705544    0.354480    0.546045\n1.45    0.676109    0.344306    0.549758\n1.50    0.649247    0.343303    0.550717\n1.55    0.624723    0.340866    0.553970\n1.60    0.599240    0.331444    0.555641\n1.65    0.572706    0.328525    0.557445\n1.70    0.551314    0.325646    0.558791\n1.75    0.519035    0.319976    0.561939\n1.80    0.504124    0.321659    0.561819\n1.85    0.479600    0.317287    0.563309\n1.90    0.454126    0.312648    0.565270\n1.95    0.436918    0.313008    0.564521\n2.00    0.410820    0.309144    0.565914\n2.05    0.392532    0.301037    0.568148\n2.10    0.375629    0.302757    0.566660\n2.15    0.355142    0.301929    0.566678\n2.20    0.341616    0.297648    0.565458\n2.25    0.320141    0.296357    0.566370\n2.30    0.308472    0.294074    0.566071\n2.35    0.296052    0.292766    0.565072\n2.40    0.277326    0.290888    0.565735\n2.45    0.268848    0.290079    0.563304\n2.50    0.259728    0.290058    0.561702"""

    data2_str_b = """0.00000000e+00    9.62087287e-01    6.33672995e-01    2.11558333e+00\n1.00000000e-02    9.59591282e-01    6.31597264e-01    2.22561211e+00    \n2.00000000e-02    9.56877248e-01    6.39291188e-01    2.17143200e+00    \n3.00000000e-02    9.56163485e-01    6.33302470e-01    2.22858303e+00    \n4.00000000e-02    9.53287915e-01    6.37142080e-01    2.21579246e+00    \n5.00000000e-02    9.48598056e-01    6.43866790e-01    2.18653431e+00    \n6.00000000e-02    9.40511963e-01    6.53833702e-01    2.15788542e+00    \n7.00000000e-02    9.33966507e-01    6.64650998e-01    2.11373904e+00    \n8.00000000e-02    9.23671398e-01    6.80529767e-01    2.03291400e+00    \n9.00000000e-02    9.09416584e-01    6.93329192e-01    1.98956943e+00    \n1.00000000e-01    8.96194904e-01    7.10835530e-01    1.91955464e+00    \n1.10000000e-01    8.75607830e-01    7.35584848e-01    1.78999254e+00    \n1.50000000e-01    7.56016706e-01    8.52338205e-01    1.19024583e+00\n2.00000000e-01    5.46856846e-01    9.35571396e-01    7.54600258e-01\n2.50000000e-01    3.72624428e-01    9.72623746e-01    4.85678751e-01\n3.00000000e-01    2.76304755e-01    9.85826594e-01    3.49417845e-01\n3.50000000e-01    1.83406970e-01    9.93813339e-01    2.29709712e-01\n4.00000000e-01    1.51306573e-01    9.96038458e-01    1.84105498e-01\n4.50000000e-01    1.28525449e-01    9.97218955e-01    1.54984444e-01\n5.00000000e-01    1.12310705e-01    9.97754020e-01    1.38690448e-01\n5.50000000e-01    1.12320687e-01    9.98070016e-01    1.28434466e-01\n6.00000000e-01    9.85604417e-02    9.98503357e-01    1.13193558e-01\n6.50000000e-01    7.80082837e-02    9.99082014e-01    8.86473840e-02\n7.00000000e-01    5.87892243e-02    9.99386288e-01    7.26273764e-02\n7.50000000e-01    6.30100386e-02    9.99418772e-01    7.04606335e-02\n8.00000000e-01    4.34105398e-02    9.99715281e-01    4.94540720e-02\n8.50000000e-01    6.09791186e-02    9.99462182e-01    6.79811237e-02\n9.00000000e-01    5.06652669e-02    9.99688166e-01    5.16055345e-02\n9.50000000e-01    3.77160761e-02    9.99757083e-01    4.57594707e-02\n1.00000000e+00    5.43442389e-02    9.99666083e-01    5.35818882e-02\n1.05000000e+00    5.08573037e-02    9.99703680e-01    5.03480247e-02\n1.10000000e+00    4.06345621e-02    9.99755825e-01    4.56912003e-02\n1.15000000e+00    3.58754652e-02    9.99654623e-01    5.45266570e-02\n1.20000000e+00    2.76163938e-02    9.99782252e-01    4.32624717e-02\n1.25000000e+00    2.99704204e-02    9.99827128e-01    3.84696851e-02\n1.30000000e+00    4.64265143e-02    9.99787819e-01    4.26406491e-02\n1.35000000e+00    2.64898588e-02    9.99816091e-01    3.97571959e-02\n1.40000000e+00    2.86276224e-02    9.99830365e-01    3.81041771e-02\n1.45000000e+00    2.74635548e-02    9.99825369e-01    3.86252837e-02\n1.50000000e+00    4.81117840e-02    9.99777412e-01    4.35889971e-02\n1.55000000e+00    2.63380661e-02    9.99864186e-01    3.42245215e-02\n1.60000000e+00    1.72290395e-02    9.99902223e-01    2.88187338e-02\n1.65000000e+00    4.13479480e-02    9.99822462e-01    3.90488547e-02\n1.70000000e+00    1.70601015e-02    9.99875004e-01    3.26419530e-02\n1.75000000e+00    9.31077328e-03    9.99893991e-01    3.00612699e-02\n1.80000000e+00    4.14317722e-02    9.99837741e-01    3.73162125e-02\n1.85000000e+00    2.31414618e-02    9.99911214e-01    2.75103580e-02\n1.90000000e+00    2.20148039e-02    9.99878658e-01    3.21925153e-02\n1.95000000e+00    7.23779819e-03    9.99831332e-01    3.79194587e-02\n2.00000000e+00    4.58273043e-02    9.99865731e-01    3.38829832e-02\n2.05000000e+00    1.45900982e-02    9.99934098e-01    2.37382691e-02\n2.10000000e+00    4.18545641e-02    9.98569430e-01    3.49745321e-02\n2.15000000e+00    7.81973831e-03    9.99884583e-01    3.14530002e-02\n2.20000000e+00    4.42141471e-03    9.99884370e-01    3.15675160e-02\n2.25000000e+00    5.39133606e-03    9.99869941e-01    3.33570377e-02\n2.30000000e+00    1.56390746e-02    9.99899192e-01    2.93671752e-02\n2.35000000e+00    1.38314851e-02    9.99903444e-01    2.87090544e-02\n2.40000000e+00    6.29023767e-03    9.99852106e-01    3.56255381e-02\n2.45000000e+00    2.62867748e-02    9.99894755e-01    3.00759068e-02\n2.50000000e+00    1.33468008e-02    9.99899910e-01    2.92293440e-02"""

    df1_b = pd.read_csv(io.StringIO(data1_str_b), sep=r'\s+', header=None, names=['D', 'Q_norm', 'Psi6', 'M'])
    df2_b = pd.read_csv(io.StringIO(data2_str_b), sep=r'\s+', header=None, names=['D', 'Psi6_k', 'S_norm', 'CV_theta'])

    df1_b['D'] = df1_b['D'].round(2)
    df2_b['D'] = df2_b['D'].round(2)
    df_all_b = pd.merge(df1_b, df2_b, on='D', how='outer').sort_values('D').reset_index(drop=True)
    
    max_d_b = 2.5

    D_arr_b = df_all_b['D'].values
    Q_val_b = df_all_b['Q_norm'].values
    Psi6_r_val_b = df_all_b['Psi6'].values
    Psi6_k_val_b = df_all_b['Psi6_k'].values
    S_norm_val_b = df_all_b['S_norm'].values

    sigma_smooth_b = 1.0
    sigma_deriv_b = 0.8
    dQ_b = np.abs(np.gradient(gaussian_filter1d(Q_val_b, sigma=sigma_smooth_b), D_arr_b))
    dPsi6_k_b = np.abs(np.gradient(gaussian_filter1d(Psi6_k_val_b, sigma=sigma_smooth_b), D_arr_b))

    sm_dQ_b = gaussian_filter1d(dQ_b, sigma=sigma_deriv_b)
    sm_dPsi6_k_b = gaussian_filter1d(dPsi6_k_b, sigma=sigma_deriv_b)

    Dc1_idx_b = np.argmax(sm_dPsi6_k_b)
    Dc1_b = D_arr_b[Dc1_idx_b]
    
    glass_mask_b = (D_arr_b > Dc1_b) & (Q_val_b >= 0.96) & (S_norm_val_b > 0.96)
    
    if np.any(glass_mask_b):
        Dc2_b = D_arr_b[glass_mask_b][0]
        after_glass_b = (D_arr_b > Dc2_b) & (~glass_mask_b)
        if np.any(after_glass_b):
            Dc3_b = D_arr_b[after_glass_b][0]
        else:
            Dc3_b = max_d_b
    else:
        Dc2_b = Dc1_b + 0.1 
        Dc3_b = max_d_b
    
    x_min_b = 1e-2
    
    ax_b.axvspan(x_min_b, Dc1_b, color='#FF9999', alpha=0.15)            
    ax_b.axvspan(Dc1_b, Dc2_b, color='#FFDD99', alpha=0.2)                
    ax_b.axvspan(Dc2_b, Dc3_b, color='#99CCFF', alpha=0.15)              
    ax_b.axvspan(Dc3_b, max_d_b, color='#CCCCCC', alpha=0.3)             
    
    ax_b.axvline(Dc1_b, color='k', linestyle='--', alpha=0.8, linewidth=1.0)
    ax_b.axvline(Dc2_b, color='k', linestyle='--', alpha=0.8, linewidth=1.0)
    ax_b.axvline(Dc3_b, color='k', linestyle='--', alpha=0.8, linewidth=1.0)

    colors_b = {'Q': COLOR_Q, 'Psi_r': COLOR_PSI, 'Psi_k': '#B22222', 'Ent': '#CC79A7'}
    l1_b = ax_b.plot(D_arr_b, Q_val_b, 's--', color=colors_b['Q'], ms=3, lw=1.0, alpha=0.6, label=r'$Q$')
    l2_b = ax_b.plot(D_arr_b, Psi6_r_val_b, 'd-.', color=colors_b['Psi_r'], mfc='white', ms=3, lw=1.0, label=r'$\Psi_6$')
    l4_b = ax_b.plot(D_arr_b, S_norm_val_b, '^-', color=colors_b['Ent'], mfc='white', ms=3, lw=1.0, label=r'$S_{n}$')

    lines_main_b = l1_b + l2_b + l4_b
    labels_main_b = [l.get_label() for l in lines_main_b]
    
    ax_b.legend(lines_main_b, labels_main_b, loc='lower center', bbox_to_anchor=(0.2, 0.98, 0.5, 0.5),
                ncol=3, 
                framealpha=0.85, edgecolor='none', labelspacing=0.3, handlelength=1.2)

    ax_b.set_xlabel(r'$\delta$')
    ax_b.set_ylabel(r'Norm. Obs.')
    ax_b.set_ylim(0.11, 1.1)
    ax_b.set_xscale('log')
    ax_b.set_xlim(0.025, max_d_b)
    ax_b.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    ax_b.grid(True, which='major', linestyle='-', alpha=0.3)
    ax_b.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    xcords = [0.055,0.17,0.5,1.6]
    xtext = ['SkX','SkBG','SkG','TBP']
    for ix,xcord in enumerate(xcords):
        ax_b.text(xcord, 0.22, xtext[ix], 
            color='black',       
            fontsize=FONT_SIZE,
            ha='center', va='center',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    

    # ==========================================
    # Subplot (c): Autocorrelation (Formerly d)
    # ==========================================
    
    print("Plotting (c)...")
    
    FILES_TO_PLOT_2 = [
        {"path": os.path.join(target_data_dir, "Autocorrelation_LL_Liquid_iT1.txt"), "label": "Fluid", "color": COLOR_Q, "marker": "o"}, 
        {"path": os.path.join(target_data_dir, "Autocorrelation_LL_iT0_S (2).txt"), "label": "SkG", "color": COLOR_PSI, "marker": "s"}
    ]
    
    for info in FILES_TO_PLOT_2:
        if os.path.exists(info["path"]):
            data = np.loadtxt(info["path"], skiprows=1)
            t_raw, At_raw = data[:, 0], data[:, 1]
            At_norm = At_raw / (At_raw[0] if At_raw[0] != 0 else 1)
            t_p, y_m, y_s = log_bin_statistics(t_raw, At_norm)
            
            if len(t_p) > 0:
                ax_c.fill_between(t_p, y_m-y_s, y_m+y_s, color=info['color'], alpha=0.12)
                idx = np.unique(np.logspace(0, np.log10(len(t_p)-1), 10).astype(int)).tolist()
                ax_c.plot(t_p, y_m, color=info['color'], lw=1.0, 
                          marker=info['marker'], markevery=idx, 
                          mec=info['color'], mfc='w', ms=4, 
                          label=info['label'])
        else:
            t_p = np.logspace(0, 5, 100)
            y_m = np.exp(-t_p/(100 if "Fluid" in info["label"] else 10000))
            idx = np.unique(np.logspace(0, np.log10(len(t_p)-1), 10).astype(int)).tolist()
            ax_c.plot(t_p, y_m, color=info['color'], lw=1.0, 
                      marker=info['marker'], markevery=idx, 
                      mec=info['color'], mfc='w', ms=4, 
                      label=info['label'])

    ax_c.set_xlabel(r"Time $t$ (MCS)"); ax_c.set_ylabel(r"$A(t)$", labelpad=1)
    ax_c.set_xscale('log')
    ax_c.set_ylim(-0.2, 1.2)
    ax_c.axhline(0, color='k', ls=':', alpha=0.5)
    ax_c.grid(True, which='major', linestyle='-', alpha=0.3)
    
    ax_c.legend(loc='lower left', framealpha=0.6, labelspacing=0.4, handlelength=1.5)

    # ==========================================
    # Row 2 (d)-(g): Spatial Distributions (Formerly e-h)
    # ==========================================
    print("Plotting (d)-(g)...")
    
    # 格式：(iD, iB, frame_idx) -> -1 代表抽取最后一帧，你可以按需改为 0, 1, 2...
    FILE_INDICES = [(4, 0, -1), (10, 0, 2000), (30, 0, -1), (60, 0, -1)]
    titles = [r'SkBG, $\delta=0.2$', r'SkG, $\delta=0.5$', r'TBP, $\delta=1.5$', r'TBP, $\delta=3.0$']
    
    last_qr = None
    for i, (ax_bot, (id_val, ib_val, frame_idx), tit) in enumerate(zip(axes_defg, FILE_INDICES, titles)):
        
        # 传递设定的 frame_idx 给载入函数
        pos_p, theta, phi, XI, YI, ZI, crop_size = load_spatial_data(id_val, ib_val, data_dir=target_data_dir, frame_idx=frame_idx)
    
        u, v = np.cos(phi), np.sin(phi)
        qr = ax_bot.quiver(pos_p[:, 0], pos_p[:, 1], u, v, theta, cmap=THETA_CMAP, norm=Normalize(0, np.pi), rasterized=True, **ARROW_OPTS)
        last_qr = qr
        
        xc, yc = np.mean(pos_p[:, 0]), np.mean(pos_p[:, 1])
        x_min, x_max = xc - crop_size/3, xc + crop_size/3
        y_min, y_max = yc - crop_size/2.5, yc + crop_size/2.5
        
        offset_x = crop_size / 8 
        cut_bottom_amount = crop_size / 6  
        
        ax_bot.set_xlim(x_min + offset_x, x_max + offset_x)
        ax_bot.set_ylim(y_min + cut_bottom_amount, y_max)
        ax_bot.set_aspect('equal')

        ax_bot.set_title(tit, fontsize=FONT_SIZE)

        ax_bot.set_xlabel(r'$x/a$',labelpad=0.5)
        ax_bot.set_ylabel(r'$y/a$',labelpad=0.5)
        
        # --- [MODIFIED LOGIC: Force Ticks to Start from 0] ---
        # Get the actual data limits we just set
        x0, x1 = ax_bot.get_xlim()
        y0, y1 = ax_bot.get_ylim()

        # Create intervals of 20 relative to the window size
        ticks_x = np.arange(0, x1 - x0, 20)
        ticks_y = np.arange(0, y1 - y0, 20)

        # Apply ticks offset by x0, y0, but explicitly label them starting from 0
        ax_bot.set_xticks(x0 + ticks_x)
        ax_bot.set_xticklabels([f"{int(t)}" for t in ticks_x])

        ax_bot.set_yticks(y0 + ticks_y)
        ax_bot.set_yticklabels([f"{int(t)}" for t in ticks_y])
        # -----------------------------------------------------
            
        ax_sf = ax_bot.inset_axes([0.6, 0.02, 0.38, 0.38])
        z_max = np.nanmax(ZI) if np.nanmax(ZI) > 1e-10 else 1.0
        ZI_norm = ZI / z_max 
        
        pcm_sf = ax_sf.pcolormesh(XI, YI, ZI_norm, cmap=SF_CMAP, shading='auto', norm=LogNorm(vmin=0.01, vmax=1.0), rasterized=True)
        last_sf = pcm_sf 
        
        ax_sf.set_xticks([]); ax_sf.set_yticks([])
        for spine in ax_sf.spines.values(): 
            spine.set_edgecolor('white')
            spine.set_linewidth(1.0)

    
    # 1. Main Colorbar (Polar Angle)
    if last_qr is not None:
        cax_main = fig.add_axes([0.975, 0.25+0.025, 0.012, 0.12])
        cb_main = fig.colorbar(last_qr, cax=cax_main, ticks=[0, np.pi/2, np.pi])
        cb_main.ax.set_yticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$'])
        cb_main.set_label(r'$\theta$', labelpad=4, fontsize=FONT_SIZE)
        cb_main.ax.tick_params(labelsize=FONT_SIZE-1)

    # 2. Inset Colorbar (Structure Factor)
    if 'last_sf' in locals() and last_sf is not None:
        cax_sf = fig.add_axes([0.975, 0.08+0.025, 0.012, 0.12]) 
        cb_sf = fig.colorbar(last_sf, cax=cax_sf, ticks=[0.01, 0.1, 1.0])
        cb_sf.ax.set_yticklabels(['0.01', '0.1', '1'])
        cb_sf.set_label(r'$S(\mathbf{q})/S_{\mathrm{max}}$', labelpad=4, fontsize=FONT_SIZE)
    
    out_name = "Fig2.pdf"
    plt.savefig(out_name, bbox_inches='tight')
    plt.savefig('Fig2.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Successfully generated and saved: {out_name}")
    plt.show()

if __name__ == "__main__":
    generate_master_figure()

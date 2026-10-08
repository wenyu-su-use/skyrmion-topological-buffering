# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import matplotlib.patheffects as path_effects
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from scipy.interpolate import griddata
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import os

# ==========================================
# 字体大小统一控制区
# ==========================================
FONT_SIZE_SCHEMATIC = 16
FONT_SIZE_DATA_PANEL_LETTER = 16  
FONT_SIZE_DATA_TITLE = 16         
FONT_SIZE_DATA_AXES_LABEL = 16    
FONT_SIZE_DATA_TICK = 16           
FONT_SIZE_DATA_LEGEND = 16        

# ==========================================
# 全局样式与配置 (RevTex-4 风格)
# ==========================================
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': FONT_SIZE_DATA_AXES_LABEL,      
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'axes.labelsize': FONT_SIZE_DATA_AXES_LABEL,        
    'axes.titlesize': FONT_SIZE_DATA_TITLE,        
    'xtick.labelsize': FONT_SIZE_DATA_TICK,        
    'ytick.labelsize': FONT_SIZE_DATA_TICK,
    'legend.fontsize': FONT_SIZE_DATA_LEGEND,
    'pdf.fonttype': 42
})

try:
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}'
except:
    rcParams['text.usetex'] = False

# ==========================================
# 配色与几何参数
# ==========================================
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

# ==========================================
# 通用辅助函数
# ==========================================
def draw_sphere(ax, center, radius, color=COLOR_ATOM, zorder=20):
    x, y = center
    ax.add_patch(patches.Circle((x, y), radius, facecolor=color, edgecolor='none', zorder=zorder))
    ax.add_patch(patches.Circle((x - radius*0.35, y + radius*0.35), radius*0.3,
                                 facecolor='white', alpha=0.3, zorder=zorder+1))

def add_stroke(text_obj, lw=STROKE_LW, color='white'):
    text_obj.set_path_effects([
        path_effects.Stroke(linewidth=lw, foreground=color),
        path_effects.Normal()
    ])

def get_lattice_points(nx, ny):
    points = []
    for j in range(ny):
        for i in range(nx):
            x = i * A_CONST + j * A_CONST * 0.5
            y = j * A_CONST * np.sqrt(3) / 2
            points.append(np.array([x, y]))
    return np.array(points)

def get_logical_index(gx, gy, nx):
    return gy * nx + gx

def project_3d_point(origin_2d, vec_3d, scale=1.0, z_skew=(0.3, 0.5)):
    vx, vy, vz = vec_3d
    dx = vx * scale
    dy = vy * scale
    dx += vz * z_skew[0] * scale
    dy += vz * z_skew[1] * scale
    return origin_2d + np.array([dx, dy])

def index_to_physical(i, j):
    x = i + 0.5 * j
    y = (np.sqrt(3) / 2.0) * j
    return x, y

def read_simulation_parameters(filepath):
    d_map, b_map = {}, {}
    if not os.path.exists(filepath): return d_map, b_map
    current_mode = None 
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if "Index_D" in line: current_mode = 'D'; continue
                if "Index_B" in line: current_mode = 'B'; continue
                parts = line.split()
                if len(parts) >= 2:
                    idx, val = int(parts[0]), float(parts[1])
                    if current_mode == 'D': d_map[idx] = val
                    else: b_map[idx] = val
    except: pass
    return d_map, b_map

# ==========================================
# 子图绘制函数: [0, 0] 晶格示意图
# ==========================================
def plot_large_lattice_subfig(ax, nx=3, ny=3):
    ax.set_aspect('equal')
    ax.axis('off')

    points = get_lattice_points(nx, ny)
    cx, cy = nx // 2, ny // 2
    idx_k = get_logical_index(cx, cy, nx)            
    idx_i = get_logical_index(cx, cy - 1, nx)        
    idx_j = get_logical_index(cx + 1, cy - 1, nx)    
    idx_l = get_logical_index(cx - 1, cy, nx)        
    idx_m = get_logical_index(cx - 1, cy + 1, nx)    
    
    idx_tmid = get_logical_index(nx - 2, ny - 1, nx)
    idx_tright = get_logical_index(nx - 1, ny - 1, nx)
    p_tmid = points[idx_tmid]
    p_tright = points[idx_tright]
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if np.isclose(dist, A_CONST, atol=0.05):
                ax.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 
                        color=COLOR_BOND_NEUTRAL, lw=LW_BOND, zorder=1, solid_capstyle='round')

    origin = points[0] 
    ax.annotate("", xy=origin + np.array([0.7, 0]), xytext=origin,
                arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.25", color=COLOR_BASIS, lw=1.0), zorder=5)
    t_a1 = ax.text(origin[0] + 0.35, origin[1] - 0.15, r'$\mathbf{a}_1$', color=COLOR_BASIS, fontsize=FONT_SIZE_SCHEMATIC, fontweight='bold', ha='center')
    add_stroke(t_a1)
    
    a2_vec = np.array([0.5, np.sqrt(3)/2]) * 0.7
    ax.annotate("", xy=origin + a2_vec, xytext=origin,
                arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.25", color=COLOR_BASIS, lw=1.0), zorder=5)
    t_a2 = ax.text(origin[0] + a2_vec[0]*0.4 - 0.15, origin[1] + a2_vec[1]*0.4, r'$\mathbf{a}_2$', color=COLOR_BASIS, fontsize=FONT_SIZE_SCHEMATIC, fontweight='bold', ha='right')
    add_stroke(t_a2)

    dm_cycle = [idx_i, idx_j, idx_k] 
    p_i, p_j, p_k = points[idx_i], points[idx_j], points[idx_k]
    center_dm = (p_i + p_j + p_k) / 3.0

    for idx in range(3):
        p_start = points[dm_cycle[idx]]
        p_end = points[dm_cycle[(idx + 1) % 3]]
        ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color=COLOR_CHI_GREEN, lw=LW_DM, zorder=4, solid_capstyle='round')

    labels_edge = [r'$D_{ij}$', r'$D_{jk}$', r'$D_{ki}$']
    for idx in range(3):
        p_start = points[dm_cycle[idx]]
        p_end = points[dm_cycle[(idx + 1) % 3]]
        mid = (p_start + p_end) / 2
        vec = p_end - p_start
        ax.add_patch(patches.FancyArrowPatch(
            posA=mid - vec * 0.05, posB=mid + vec * 0.05, arrowstyle=f"-|>,head_width={ARROW_HW},head_length={ARROW_HL}", color=COLOR_CHI_GREEN, lw=0, zorder=10 
        ))
        vec_out = mid - center_dm
        vec_out = vec_out / np.linalg.norm(vec_out)
        pos_label = mid + vec_out * 0.12  
        
        ha, va = 'center', 'center'
        if vec_out[1] < -0.5: va = 'top'
        elif vec_out[0] > 0: ha, va = 'left', 'center'
        else: ha, va = 'right', 'center'
        
        t_edge = ax.text(pos_label[0], pos_label[1], labels_edge[idx], ha=ha, va=va, fontsize=FONT_SIZE_SCHEMATIC, color=COLOR_CHI_GREEN, fontweight='bold')
        add_stroke(t_edge)

    inradius = A_CONST / (2 * np.sqrt(3))
    draw_radius = inradius * 0.6
    ax.add_patch(patches.Circle(center_dm, draw_radius, fill=False, edgecolor=COLOR_CHI_GREEN, lw=1.2, zorder=9))
    
    for angle_deg in [0, 120, 240]:
        angle_rad = np.radians(angle_deg)
        pos_x = center_dm[0] + draw_radius * np.cos(angle_rad)
        pos_y = center_dm[1] + draw_radius * np.sin(angle_rad)
        ax.add_patch(patches.FancyArrowPatch(
            posA=(pos_x + np.sin(angle_rad)*0.01, pos_y - np.cos(angle_rad)*0.01),
            posB=(pos_x - np.sin(angle_rad)*0.01, pos_y + np.cos(angle_rad)*0.01),
            arrowstyle=f"-|>,head_width={ARROW_HW*0.8},head_length={ARROW_HL*0.8}", color=COLOR_CHI_GREEN, lw=0, zorder=4
        ))
    t_D_vec = ax.text(center_dm[0], center_dm[1], r'$\mathbf{D}$', ha='center', va='center', fontsize=FONT_SIZE_SCHEMATIC, color=COLOR_CHI_GREEN, fontweight='bold')
    add_stroke(t_D_vec)

    ax.add_patch(patches.FancyArrowPatch(
        posA=p_tmid + np.array([0.05, 0.1]), posB=p_tright + np.array([-0.05, 0.1]),
        connectionstyle="arc3,rad=-0.35", arrowstyle="-", color=COLOR_ARROW_J, lw=1.2, zorder=10
    ))
    
    mid_j = (p_tmid + p_tright) / 2
    t_j = ax.text(mid_j[0], mid_j[1] + 0.25, r'$J$', color=COLOR_ARROW_J, ha='center', va='bottom', fontsize=FONT_SIZE_SCHEMATIC, fontweight='bold')
    add_stroke(t_j)

    p_l, p_k, p_m = points[idx_l], points[idx_k], points[idx_m]
    center_chi_2d = (p_l + p_k + p_m) / 3.0
    
    def get_inward_vector(start_pos, center_pos, z_component=0.8):
        vec_xy = center_pos - start_pos
        vec_xy = vec_xy / np.linalg.norm(vec_xy)
        vec_3d = np.array([vec_xy[0]*0.5, vec_xy[1]*0.5, z_component])
        return vec_3d / np.linalg.norm(vec_3d)

    vec_l = get_inward_vector(p_l, center_chi_2d)
    vec_k = get_inward_vector(p_k, center_chi_2d)
    vec_m = get_inward_vector(p_m, center_chi_2d)
    
    arrow_len = 0.85
    skew = (0.25, 0.45) 
    tip_l = project_3d_point(p_l, vec_l, scale=arrow_len, z_skew=skew)
    tip_k = project_3d_point(p_k, vec_k, scale=arrow_len, z_skew=skew)
    tip_m = project_3d_point(p_m, vec_m, scale=arrow_len, z_skew=skew)
    
    ax.add_patch(patches.Polygon([p_m, p_l, tip_l, tip_m], closed=True, facecolor=COLOR_FACE_ML, edgecolor='none', alpha=0.75, zorder=21))
    ax.add_patch(patches.Polygon([p_l, p_k, tip_k, tip_l], closed=True, facecolor=COLOR_FACE_LK, edgecolor='none', alpha=0.75, zorder=22))
    ax.add_patch(patches.Polygon([p_k, p_m, tip_m, tip_k], closed=True, facecolor=COLOR_FACE_KM, edgecolor='none', alpha=0.75, zorder=22))
    ax.add_patch(patches.Polygon([tip_l, tip_k, tip_m], closed=True, facecolor=COLOR_FACE_TOP, edgecolor=COLOR_VOL_EDGE, lw=0.8, alpha=0.85, zorder=23))
    
    for start, end in [(p_l, tip_l), (p_k, tip_k), (p_m, tip_m)]:
        ax.plot([start[0], end[0]], [start[1], end[1]], color=COLOR_VOL_EDGE, lw=0.8, ls='-', alpha=0.8, zorder=23)
        bg_arrow = patches.FancyArrowPatch(posA=start, posB=end, arrowstyle=f"simple,tail_width=2,head_width={ARROW_HW*1.5},head_length={ARROW_HL*1.5}", color='white', zorder=24, shrinkA=R_SPHERE*0.9, shrinkB=0.0)
        ax.add_patch(bg_arrow)
        main_arrow = patches.FancyArrowPatch(posA=start, posB=end, arrowstyle=f"simple,tail_width=1.2,head_width={ARROW_HW},head_length={ARROW_HL}", color=COLOR_SPIN_3D, zorder=25, shrinkA=R_SPHERE*0.9, shrinkB=0.0)
        ax.add_patch(main_arrow)

    center_top = (tip_l + tip_k + tip_m) / 3.0
    t_chi = ax.text(center_top[0]+0.15, center_top[1] + 0.05, r'$\chi_{kml}$', ha='center', va='bottom', fontsize=FONT_SIZE_SCHEMATIC, color=COLOR_CHI_GREEN, fontweight='bold', zorder=30)
    add_stroke(t_chi)

    for i, p in enumerate(points):
        draw_sphere(ax, p, R_SPHERE)
        
    t_k = ax.text(points[idx_k][0] + 0.05, points[idx_k][1] + 0.22, r'$\mathbf{S}_k$', ha='left', va='bottom', fontsize=FONT_SIZE_SCHEMATIC)
    t_i = ax.text(points[idx_i][0], points[idx_i][1] - 0.25, r'$\mathbf{S}_i$', ha='center', va='top', fontsize=FONT_SIZE_SCHEMATIC)
    t_j = ax.text(points[idx_j][0], points[idx_j][1] - 0.25, r'$\mathbf{S}_j$', ha='center', va='top', fontsize=FONT_SIZE_SCHEMATIC)
    t_l = ax.text(points[idx_l][0] - 0.25, points[idx_l][1] - 0.05, r'$\mathbf{S}_l$', ha='right', va='top', fontsize=FONT_SIZE_SCHEMATIC)
    t_m = ax.text(points[idx_m][0] - 0.15, points[idx_m][1] + 0.25, r'$\mathbf{S}_m$', ha='center', va='bottom', fontsize=FONT_SIZE_SCHEMATIC)
    for t in [t_k, t_i, t_j, t_l, t_m]: add_stroke(t)

    max_x = (nx - 1) * A_CONST + (ny - 1) * A_CONST * 0.5
    max_y = (ny - 1) * A_CONST * np.sqrt(3) / 2
    ax.set_xlim(-0.35, max_x + 0.35)
    ax.set_ylim(-0.40, max_y + 0.35)

# ==========================================
# 子图绘制函数: [1, 0] 物理量折线图
# ==========================================
def load_observables_data():
    _, b_map = read_simulation_parameters("../datas_D0/Scan_Parameters_D0.txt")
    psi6_file = "../datas_D0/Psi6_vs_B_filtered_data_D0.txt"
    iD = 0 
    
    b_plot, m_plot, m_err, q_plot, q_err = [], [], [], [], []
    has_psi6 = False
    B_psi, V_psi, E_psi = [], [], []

    if b_map:
        b_list = [b_map[idx] for idx in sorted(b_map.keys())]
        for iB, b_val in enumerate(b_list):
            fname = f"../datas_D0/Averaged_Measures_iD{iD}_iB{iB}.txt"
            if not os.path.exists(fname): continue
            try:
                data = np.loadtxt(fname)
                b_plot.append(b_val)
                m_plot.append(data[0, 1])     
                m_err.append(data[1, 1])
                q_plot.append(abs(data[0, 3])) 
                q_err.append(data[1, 3])      
            except: pass

        if os.path.exists(psi6_file):
            psi_d = np.loadtxt(psi6_file)
            B_psi, V_psi, E_psi = psi_d[:, 0], psi_d[:, 1], psi_d[:, 2]
            has_psi6 = True
            
    if len(b_plot) == 0:
        B = np.linspace(0, 1.2, 12)
        M = 1 / (1 + np.exp(-10 * (B - 0.6)))
        M_err = np.random.uniform(0.01, 0.05, len(B))
        Q_norm = np.exp(-((B - 0.6)**2) / 0.05)
        Q_err_norm = np.random.uniform(0.02, 0.08, len(B))
        B_psi, V_psi, E_psi = B, 1 - Q_norm, M_err
        has_psi6 = True
    else:
        q_scale = np.max(q_plot) if np.max(q_plot) != 0 else 1.0
        B, M, M_err = np.array(b_plot), np.array(m_plot), np.array(m_err)
        Q_norm = np.array(q_plot) / q_scale
        Q_err_norm = np.array(q_err) / q_scale 

    return B, M, M_err, Q_norm, Q_err_norm, B_psi, V_psi, E_psi, has_psi6

def plot_observables_subfig(ax):
    B, M, M_err, Q_norm, Q_err_norm, B_psi, V_psi, E_psi, has_psi6 = load_observables_data()

    ax.errorbar(B, Q_norm, yerr=Q_err_norm, fmt='s--', color=COLOR_Q, 
                markersize=4, elinewidth=1.0, capsize=2, alpha=0.6, label=r'$Q/Q_{\rm max}$')

    if has_psi6:
        ax.errorbar(B_psi, V_psi, yerr=E_psi, fmt='d-.', color=COLOR_PSI, 
                    markersize=4.5, elinewidth=1.0, capsize=2, mfc='white', label=r'$\Psi_6$')

    ax.set_xlabel(r'Magnetic Field $B$', fontsize=FONT_SIZE_DATA_AXES_LABEL)
    ax.set_ylabel(r'Normalized Observables', fontsize=FONT_SIZE_DATA_AXES_LABEL)
    ax.set_xlim(0, np.max(B) * 1.05)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.08), fontsize=FONT_SIZE_DATA_LEGEND)

# ==========================================
# 小子图 (磁场/结构因子) 数据生成
# ==========================================
blue_coolwarm = plt.get_cmap('coolwarm')(0.0) 
red_rdbu      = plt.get_cmap('bwr')(1.0)   
mid_white     = '#FFFFFF' 
hybrid_colors = [blue_coolwarm, mid_white, red_rdbu]
HYBRID_CMAP = LinearSegmentedColormap.from_list('MyHybridCmap', hybrid_colors)

ARROW_OPTS = {
    'scale': 25,
    'scale_units': 'inches',
    'width': 0.005,
    'headwidth': 3.5,
    'headlength': 3,
    'minlength': 3,
    'alpha': 1.0,
    'pivot': 'middle'
}
THETA_CMAP = HYBRID_CMAP
SF_CMAP = 'hot'

TARGET_D_SMALL_PLOTS = 0
PARAM_PAIRS_SMALL_PLOTS = [(TARGET_D_SMALL_PLOTS, 0), (TARGET_D_SMALL_PLOTS, 1), (TARGET_D_SMALL_PLOTS, 2), 
                           (TARGET_D_SMALL_PLOTS, 3), (TARGET_D_SMALL_PLOTS, 4), (TARGET_D_SMALL_PLOTS, 5)]
Lx, Ly = 90, 90
PARAM_FILE = '../dataD0/Scan_Parameters.txt'
PLOT_LIMIT = 2.3
GRID_RES = 600  
CROP_SIZE = 60 

TARGET_FRAME_IDX = 3000

def load_data_small_plots(id_val, ib_val, frame_idx=-1):
    I, J = np.meshgrid(np.arange(Lx), np.arange(Ly))
    X_phys, Y_phys = index_to_physical(I, J)
    pos_phys = np.vstack((X_phys.flatten(), Y_phys.flatten())).T
    
    f_theta = '../dataD0/allbins_theta_iD%d_iB%d.bin' % (id_val, ib_val)
    f_phi = '../dataD0/allbins_phi_iD%d_iB%d.bin' % (id_val, ib_val)
    
    N_spins = Lx * Ly 
    
    if os.path.exists(f_theta) and os.path.exists(f_phi):
        data_theta_all = np.fromfile(f_theta, dtype=float)
        data_phi_all = np.fromfile(f_phi, dtype=float)
        
        total_frames = len(data_theta_all) // N_spins
        
        if total_frames > 0:
            if frame_idx < 0:
                frame_idx = total_frames + frame_idx
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            start_idx = frame_idx * N_spins
            end_idx = start_idx + N_spins
            
            theta = data_theta_all[start_idx:end_idx]
            phi = data_phi_all[start_idx:end_idx]
            print(f"Loaded config {frame_idx + 1}/{total_frames} from iD={id_val}, iB={ib_val}")
        else:
            print(f"Warning: {f_theta} is empty!")
            theta = np.zeros(N_spins)
            phi = np.zeros(N_spins)
    else:
        theta = np.abs(np.sin(X_phys/4)*np.cos(Y_phys/4)*np.pi).flatten()
        phi = np.arctan2(Y_phys-np.mean(Y_phys), X_phys-np.mean(X_phys)).flatten()

    f_sf = "../dataD0/SF_bin0_iD%d_iB%d.txt" % (id_val, ib_val)
    if os.path.exists(f_sf):
        data = np.loadtxt(f_sf, skiprows=1)
        qx, qy, sq = data[:, 0], data[:, 1], data[:, 2]
        xi = yi = np.linspace(-PLOT_LIMIT, PLOT_LIMIT, GRID_RES)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((qx, qy), sq, (XI, YI), method='linear', fill_value=1e-10)
    else:
        XI, YI = np.meshgrid(np.linspace(-PLOT_LIMIT, PLOT_LIMIT, GRID_RES), 
                             np.linspace(-PLOT_LIMIT, PLOT_LIMIT, GRID_RES))
        ZI = np.exp(-(XI**2 + YI**2)/0.1)
        for ang in np.linspace(0, 2*np.pi, 7):
            ZI += 0.5 * np.exp(-((XI-1.5*np.cos(ang))**2 + (YI-1.5*np.sin(ang))**2)/0.08)
        
    return pos_phys, theta, phi, XI, YI, ZI

# ==========================================
# 主函数：2行 x 5列
# ==========================================
def plot_combined_figure():
    fig = plt.figure(figsize=(16, 7)) 

    gs = fig.add_gridspec(2, 5, width_ratios=[1.3, 1, 1, 1, 0.05], 
                          left=0.04, right=0.92, bottom=0.1, top=0.9, 
                          wspace=0.15, hspace=0.25)

    ax_big = fig.add_subplot(gs[0, 0])
    plot_large_lattice_subfig(ax_big)
    ax_big.text(-0.02, 1.1, '(a)', transform=ax_big.transAxes, fontsize=FONT_SIZE_DATA_PANEL_LETTER, fontweight='bold')

    ax_obs_container = fig.add_subplot(gs[1, 0])
    ax_obs_container.axis('off')
    ax_obs = ax_obs_container.inset_axes([0.23, 0.0, 0.77, 1.0])
    plot_observables_subfig(ax_obs)
    ax_obs.set_box_aspect(1)
    ax_obs.text(-0.25, 1.08, '(b)', transform=ax_obs.transAxes, fontsize=FONT_SIZE_DATA_PANEL_LETTER, fontweight='bold')

    d_map_small, b_map_small = read_simulation_parameters(PARAM_FILE)
    small_plot_labels = ['(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    # ==========================================
    # 【核心修改】数据预读取与单图归一化 (Normalization)
    # ==========================================
    plot_data_cache = []
    
    for id_val, ib_val in PARAM_PAIRS_SMALL_PLOTS:
        pos_phys, theta, phi, XI, YI, ZI = load_data_small_plots(id_val, ib_val, TARGET_FRAME_IDX)
        
        # 技巧：将每个图的强度 ZI 除以其本身的最大值
        z_max = np.nanmax(ZI)
        if z_max > 0:
            ZI_norm = ZI / z_max
        else:
            ZI_norm = ZI
            
        plot_data_cache.append((pos_phys, theta, phi, XI, YI, ZI_norm))

    # 设定全局相对强度 Colorbar：所有图的最大值都被缩放到了 1.0
    # LogNorm(0.01, 1.0) 能让极值呈现最亮色，同时保证低于 1% 最大强度的部分才融入背景黑/深色中。
    global_norm_sf = LogNorm(vmin=0.01, vmax=1.0)
    # ==========================================

    for i, (id_val, ib_val) in enumerate(PARAM_PAIRS_SMALL_PLOTS):
        row = i // 3
        col = (i % 3) + 1  
        
        # 读取已经归一化好的数据 ZI_norm
        pos_phys, theta, phi, XI, YI, ZI_norm = plot_data_cache[i]
        
        real_D = d_map_small.get(id_val, id_val * 0.1)
        real_B = b_map_small.get(ib_val, ib_val * 0.1)
        
        ax_small_main = fig.add_subplot(gs[row, col])
        u, v = np.cos(phi), np.sin(phi)
        qr_small = ax_small_main.quiver(pos_phys[:, 0], pos_phys[:, 1], u, v, theta,
                                         cmap=THETA_CMAP, norm=Normalize(0, np.pi), **ARROW_OPTS)
        
        xc_phys, yc_phys = np.mean(pos_phys[:, 0]), np.mean(pos_phys[:, 1])
        ax_small_main.set_xlim(xc_phys - CROP_SIZE/2, xc_phys + CROP_SIZE/2)
        ax_small_main.set_ylim(yc_phys - CROP_SIZE/2, yc_phys + CROP_SIZE/2)
        ax_small_main.set_aspect('equal')
        
        ax_small_main.set_title(r'$B = %.2f, \delta = %.2f$' % (real_B, real_D), pad=12, fontsize=FONT_SIZE_DATA_TITLE)
        ax_small_main.text(-0.15, 1.08, small_plot_labels[i], transform=ax_small_main.transAxes, fontsize=FONT_SIZE_DATA_PANEL_LETTER, fontweight='bold')
        
        if row == 1: 
            ax_small_main.set_xlabel(r'$x / a$', fontsize=FONT_SIZE_DATA_AXES_LABEL, labelpad=0.5)
        else:
            ax_small_main.set_xticklabels([])
            
        if col == 1: 
            ax_small_main.set_ylabel(r'$y / a$', fontsize=FONT_SIZE_DATA_AXES_LABEL, labelpad=0.5)
        else:
            ax_small_main.set_yticklabels([])

        ax_small_inset = ax_small_main.inset_axes([0.6, 0.02, 0.38, 0.38])
        
        # 使用统一的 global_norm_sf 和归一化后的 ZI_norm 画图
        im_small = ax_small_inset.pcolormesh(XI, YI, ZI_norm, cmap=SF_CMAP, shading='auto',
                                              norm=global_norm_sf, rasterized=True)
        ax_small_inset.set_aspect('equal')
        ax_small_inset.set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
        ax_small_inset.set_ylim(-PLOT_LIMIT, PLOT_LIMIT)
        ax_small_inset.set_xticks([]); ax_small_inset.set_yticks([])
        
        for spine in ax_small_inset.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(1.0)

    # --- 第 4 列: 颜色条 ---
    cax_r_small = fig.add_subplot(gs[0, 4])
    cbar_r_small = fig.colorbar(qr_small, cax=cax_r_small)
    cbar_r_small.set_label(r'Polar Angle $\theta$', labelpad=8, fontsize=FONT_SIZE_DATA_AXES_LABEL)
    cbar_r_small.set_ticks([0, np.pi/2, np.pi])
    cbar_r_small.set_ticklabels(['0', r'$\pi/2$', r'$\pi$'])
    cbar_r_small.ax.tick_params(labelsize=FONT_SIZE_DATA_TICK)
    
    cax_s_small = fig.add_subplot(gs[1, 4])
    cbar_s_small = fig.colorbar(im_small, cax=cax_s_small)
    # 修改 Label 名字，告诉读者这画的是针对各子图独立归一化后的相对强度
    cbar_s_small.set_label(r'Relative $S(\mathbf{q}) / S_{\rm max}$', labelpad=8, fontsize=FONT_SIZE_DATA_AXES_LABEL)
    cbar_s_small.ax.tick_params(labelsize=FONT_SIZE_DATA_TICK)

    plt.savefig('Fig1.pdf', bbox_inches='tight')
    plt.savefig('Fig1.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_combined_figure()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_perfect_mpc_chart():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis('off')

    Z_GROUP = 1   # Garis putus-putus (paling belakang)
    Z_LINE = 2    # Garis penghubung
    Z_SHADOW = 3  # Bayangan
    Z_BOX = 4     # Kotak utama
    Z_TEXT = 5    # Teks
    Z_ARROW = 6   # Kepala Panah (Paling depan agar tegas)

    def add_box(x, y, w, h, text, color="#e0e0e0", fontsize=10):
        shadow = patches.Rectangle((x+0.005, y-0.005), w, h, facecolor='gray', alpha=0.3, zorder=Z_SHADOW)
        ax.add_patch(shadow)
        rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=2, zorder=Z_BOX)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, fontweight='bold', zorder=Z_TEXT)
        return {'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + w/2, 'cy': y + h/2, 
                'top': y+h, 'bottom': y, 'left': x, 'right': x+w}

    def add_group_label(x, y, w, h, label):
        rect = patches.Rectangle((x, y), w, h, facecolor="none", edgecolor="#000080", 
                                 linestyle="--", linewidth=1.5, zorder=Z_GROUP)
        ax.add_patch(rect)
        ax.text(x + w/2, y - 0.025, label, ha="center", va="top", fontsize=11, 
                style='italic', fontweight='bold', color="#000080", zorder=Z_TEXT)

    def draw_arrow_line(p1, p2, method='Direct', double=False):
        """
        Menggambar garis dengan Kepala Panah Segitiga Solid (Marker).
        """
        x1, y1 = p1
        x2, y2 = p2
        xs, ys = [], []
        final_dx, final_dy = 0, 0 # Arah segmen terakhir (untuk kepala panah)
        start_dx, start_dy = 0, 0 # Arah segmen pertama (untuk panah ganda)

        if method == 'HV': # Horizontal -> Vertical
            xs = [x1, x2, x2]
            ys = [y1, y1, y2]
            final_dx, final_dy = 0, y2 - y1
            start_dx, start_dy = x2 - x1, 0
        elif method == 'VH': # Vertical -> Horizontal
            xs = [x1, x1, x2]
            ys = [y1, y2, y2]
            final_dx, final_dy = x2 - x1, 0
            start_dx, start_dy = 0, y2 - y1
        else: # Direct
            xs = [x1, x2]
            ys = [y1, y2]
            final_dx, final_dy = x2 - x1, y2 - y1
            start_dx, start_dy = x2 - x1, y2 - y1
        ax.plot(xs, ys, color="black", lw=2, zorder=Z_LINE)
        head_marker = ''
        if abs(final_dx) > abs(final_dy): # Gerak Horizontal
            head_marker = '>' if final_dx > 0 else '<'
        else: # Gerak Vertikal
            head_marker = '^' if final_dy > 0 else 'v'
        ax.plot(x2, y2, marker=head_marker, color='black', markersize=12, 
                markeredgecolor='black', zorder=Z_ARROW)
        if double:
            start_marker = ''
            if abs(start_dx) > abs(start_dy):
                start_marker = '<' if start_dx > 0 else '>' 
            else:
                start_marker = 'v' if start_dy > 0 else '^'
            
            ax.plot(x1, y1, marker=start_marker, color='black', markersize=12, 
                    markeredgecolor='black', zorder=Z_ARROW)

    #Blok


    add_group_label(0.05, 0.68, 0.25, 0.22, "Field Instrumentation")
    box_sensor = add_box(0.08, 0.74, 0.19, 0.1, "Geothermal Sensors\n(T, P, Flow)")
    add_group_label(0.05, 0.15, 0.25, 0.45, "Data Pre-processing")
    box_filter = add_box(0.08, 0.42, 0.19, 0.1, "Noise Filter\n& Calibration")
    box_estim = add_box(0.08, 0.22, 0.19, 0.1, "State Estimator\n(Observer)")
    add_group_label(0.35, 0.25, 0.30, 0.40, "Main Controller Unit")
    box_mpc = add_box(0.38, 0.35, 0.24, 0.2, "AI - MPC Core\n(Optimization Algorithm)", color="#fff2cc", fontsize=11)
    shift_x_group = 0.75
    shift_x_box = 0.78
    add_group_label(shift_x_group, 0.68, 0.25, 0.22, "Actuation System")
    box_valve = add_box(shift_x_box, 0.74, 0.19, 0.1, "Control Valves\n(Steam Input)")
    add_group_label(shift_x_group, 0.15, 0.25, 0.45, "Digital Interface")
    box_cloud = add_box(shift_x_box, 0.42, 0.19, 0.1, "Cloud Database\n& Historian")
    box_twin = add_box(shift_x_box, 0.22, 0.19, 0.1, "Digital Twin\nSimulation")
    add_group_label(0.35, 0.05, 0.30, 0.15, "Display Information")
    box_disp = add_box(0.38, 0.08, 0.24, 0.08, "Operator Dashboard")

# Koneksi Panah

    draw_arrow_line((box_sensor['cx'], box_sensor['bottom']), 
                    (box_filter['cx'], box_filter['top']), method='Direct')
    draw_arrow_line((box_filter['cx'], box_filter['bottom']), 
                    (box_estim['cx'], box_estim['top']), method='Direct')
    p_est = (box_estim['right'], box_estim['cy'])
    p_mpc_in = (box_mpc['left'], box_mpc['cy'])
    mid_x = (p_est[0] + p_mpc_in[0]) / 2 - 0.05
    ax.plot([p_est[0], mid_x, mid_x, p_mpc_in[0]], 
            [p_est[1], p_est[1], p_mpc_in[1], p_mpc_in[1]], 
            color="black", lw=2, zorder=Z_LINE)
    ax.plot(p_mpc_in[0], p_mpc_in[1], marker='>', color='black', markersize=12, zorder=Z_ARROW)
    start_pt = (box_mpc['right'], box_mpc['cy'])
    end_pt = (box_valve['left'], box_valve['cy'])
    mid_gap_x = (start_pt[0] + end_pt[0]) / 2
    ax.plot([start_pt[0], mid_gap_x, mid_gap_x, end_pt[0]], 
            [start_pt[1], start_pt[1], end_pt[1], end_pt[1]], 
            color="black", lw=2, zorder=Z_LINE)
    ax.plot(end_pt[0], end_pt[1], marker='>', color='black', markersize=12, zorder=Z_ARROW)
    draw_arrow_line((box_mpc['right'], box_cloud['cy']), 
                    (box_cloud['left'], box_cloud['cy']), method='Direct', double=True)
    draw_arrow_line((box_cloud['cx'], box_cloud['bottom']), 
                    (box_twin['cx'], box_twin['top']), method='Direct', double=True)
    draw_arrow_line((box_mpc['cx'], box_mpc['bottom']), 
                    (box_disp['cx'], box_disp['top']), method='Direct')
    # Figur
    plt.text(0.5, 0.95, "Figure 5. Hardware & Process Flow Diagram (AI-MPC Integrated)", 
             ha="center", fontsize=14, fontweight='bold', zorder=Z_TEXT)

    plt.tight_layout()
    plt.savefig('mpc_final_perfect_arrow.png', dpi=300)
    plt.show()

draw_perfect_mpc_chart()
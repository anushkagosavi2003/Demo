import gradio as gr
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -------------------------------
# Load YOLO model
# -------------------------------

model = YOLO("best.pt")

# -------------------------------
# Helpers
# -------------------------------

def get_video_path(video):
    if video is None:
        return None
    if isinstance(video, dict):
        return video["video"]
    return video


def get_first_frame(video):
    video_path = get_video_path(video)
    if video_path is None:
        return None
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def reset_points():
    return {"start": None, "points": []}


def record_click(evt: gr.SelectData, state, image):
    x, y = evt.index
    img = image.copy()

    if state["start"] is None:
        state["start"] = (x, y)
        return state, img

    x1, y1 = state["start"]
    x2, y2 = x, y

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dx > dy:
        y2 = y1
    else:
        x2 = x1

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    state["points"] = [(x1, y1), (x2, y2)]
    state["start"] = None

    return state, img


# -------------------------------
# Calibration
# -------------------------------

def compute_scale(state, ref_m):
    points = state["points"]
    if len(points) < 2:
        return None
    (x1, y1), (x2, y2) = points
    pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    px_per_m = pixel_dist / ref_m
    pixel_to_m = 1 / px_per_m
    print(f"Calibration: {px_per_m:.2f} px/m  |  pixel_to_m = {pixel_to_m:.6f}")
    return pixel_to_m


# -------------------------------
# Video Processing — collect positions
# -------------------------------

def process_video(video, pixel_to_m):
    video_path  = get_video_path(video)
    cap         = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS)
    dt          = 1 / fps
    frame_count = 0
    initial_y   = None
    times       = []
    positions   = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if initial_y is None:
                    initial_y = cy

                position_m = (cy - initial_y) * pixel_to_m
                positions.append(position_m)
                times.append(frame_count * dt)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # Small font anchored top-left — never clipped
                label = f"y={position_m:.4f}m"
                cv2.putText(
                    frame, label,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0), 1, cv2.LINE_AA
                )

        frame_count += 1
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), times, positions

    cap.release()


# -------------------------------
# Save position data to Excel
# -------------------------------

def save_position_to_excel(times, positions, output_file="position_data.xlsx"):
    df = pd.DataFrame({"Time (s)": times, "Position (m)": positions})
    df.to_excel(output_file, index=False)
    print(f"✅ Position data saved to {output_file}")


# -------------------------------
# Physics Functions
# -------------------------------

def experimental_cd(dp, rho_p, rho_f, vt, g=9.81):
    return (4 * g * dp * (rho_p - rho_f)) / (3 * rho_f * vt ** 2)


def compute_viscosity(rho_f, vt, dp, Cd_exp):
    Re = 24.0 / Cd_exp
    mu = (rho_f * vt * dp) / Re
    return Re, mu


def flow_regime(Re):
    if Re < 1:      return "Stokes flow"
    elif Re < 2000: return "Laminar  ⚠️ (Re > 1, viscosity estimate less accurate)"
    elif Re < 4000: return "Transitional  ❌ (Stokes assumption invalid)"
    else:           return "Turbulent  ❌ (Stokes assumption invalid)"


# -------------------------------
# Equation / R² helpers
# -------------------------------

def build_equation(coeff, degree):
    terms = []
    for i, c in enumerate(coeff):
        power = degree - i
        if abs(c) < 1e-8:
            continue
        sign = "+" if c >= 0 else "-"
        val  = abs(c)
        if power == 0:
            terms.append(f"{sign} {val:.4f}")
        elif power == 1:
            terms.append(f"{sign} {val:.4f}x")
        else:
            terms.append(f"{sign} {val:.4f}x$^{{{power}}}$")
    return "y = " + " ".join(terms).lstrip("+ ").strip()


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

POLY_DEGREE = 5   # hardcoded — no slider needed
def create_plots(times, positions, poly_degree=POLY_DEGREE):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    times     = np.array(times)
    positions = np.array(positions)

    # Clean data
    df      = pd.DataFrame({"t": times, "y": positions})
    df      = df.groupby("t", as_index=False)["y"].mean().sort_values("t")
    t_clean = df["t"].values
    y_clean = df["y"].values

    # Normalize time
    t0      = t_clean[0]
    t_range = t_clean[-1] - t0
    t_norm  = (t_clean - t0) / t_range

    # Polynomial fit
    coeff    = np.polyfit(t_norm, y_clean, poly_degree)
    poly_pos = np.poly1d(coeff)

    t_s      = np.linspace(0, 1, 1000)
    pos_fit  = poly_pos(t_s)
    t_actual = t_s * t_range + t0

    # Velocity
    poly_vel = poly_pos.deriv()
    vel_fit  = poly_vel(t_s) / t_range

    # Metrics
    r2     = r2_score(y_clean, poly_pos(t_norm))
    eq_str = build_equation(coeff, poly_degree)

    # Terminal velocity
    steady_start = int(len(vel_fit) * 0.80)
    vt = float(np.mean(vel_fit[steady_start:]))

    # ─────────────────────────────────────────────
    # Plot 1: Distance vs Time
    # ─────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    fig1.patch.set_facecolor("#f8fafc")
    ax1.set_facecolor("#f8fafc")

    step = max(1, len(t_clean) // 25)

    ax1.scatter(
        t_clean[::step], y_clean[::step],
        s=45, color="#3b82f6", alpha=0.85,
        label="Measured Position", zorder=3
    )

    ax1.plot(
        t_actual, pos_fit,
        "-", linewidth=2.5, color="#dc2626",
        label=f"Polynomial Fit (deg {poly_degree})",
        zorder=4
    )

    # Equation box
    ax1.text(
        0.5, 0.97,
        f"{eq_str}\nR² = {r2:.4f}",
        transform=ax1.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.95
        )
    )

    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Distance (m)", fontsize=11, labelpad=15)

    # Push y-label slightly left (key fix)
    ax1.yaxis.set_label_coords(-0.08, 0.5)

    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(alpha=0.25, linestyle="--")

    # Layout fix
    fig1.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.15)


    # ─────────────────────────────────────────────
    # Plot 2: Velocity vs Time
    # ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    fig2.patch.set_facecolor("#f8fafc")
    ax2.set_facecolor("#f8fafc")

    ax2.plot(
        t_actual, vel_fit,
        linewidth=2.5, color="#dc2626",
        label="Velocity (dy/dt)"
    )

    ax2.axhline(
        vt, linestyle=":", linewidth=2.5, color="black",
        label=f"Terminal Velocity = {vt:.4f} m/s"
    )

    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Velocity (m/s)", fontsize=11, labelpad=15)

    # Push y-label left
    ax2.yaxis.set_label_coords(-0.08, 0.5)

    ax2.legend(framealpha=0.9)
    ax2.grid(alpha=0.25, linestyle="--")

    # Layout fix
    fig2.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.15)

    return fig1, fig2, vt

# def create_plots(times, positions, poly_degree=POLY_DEGREE):
#     times     = np.array(times)
#     positions = np.array(positions)

#     df      = pd.DataFrame({"t": times, "y": positions})
#     df      = df.groupby("t", as_index=False)["y"].mean().sort_values("t")
#     t_clean = df["t"].values
#     y_clean = df["y"].values

#     t0      = t_clean[0]
#     t_range = t_clean[-1] - t0
#     t_norm  = (t_clean - t0) / t_range

#     coeff    = np.polyfit(t_norm, y_clean, poly_degree)
#     poly_pos = np.poly1d(coeff)

#     t_s      = np.linspace(0, 1, 1000)
#     pos_fit  = poly_pos(t_s)
#     t_actual = t_s * t_range + t0

#     poly_vel = poly_pos.deriv()
#     vel_fit  = poly_vel(t_s) / t_range

#     r2     = r2_score(y_clean, poly_pos(t_norm))
#     eq_str = build_equation(coeff, poly_degree)

#     steady_start = int(len(vel_fit) * 0.80)
#     vt = float(np.mean(vel_fit[steady_start:]))

    # # ── Plot 1: Distance vs Time ──────────────────────────────────────
    # fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    # fig1.patch.set_facecolor("#f8fafc")
    # ax1.set_facecolor("#f8fafc")

    # step = max(1, len(t_clean) // 25)
    # ax1.scatter(
    #     t_clean[::step], y_clean[::step],
    #     s=45, color="#3b82f6", alpha=0.85, zorder=3,
    #     label="Measured Position (sampled)"
    # )
    # ax1.plot(
    #     t_actual, pos_fit,
    #     "-", linewidth=2.5, color="#dc2626", zorder=4,
    #     label=f"Polynomial Fit  (degree {poly_degree})"
    # )
    # ax1.text(
    #     0.5, 0.97,
    #     f"{eq_str}\nR² = {r2:.4f}",
    #     transform=ax1.transAxes, fontsize=8.5,
    #     verticalalignment="top", horizontalalignment="center",
    #     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
    #               edgecolor="#cccccc", alpha=0.95)
    # )
    # ax1.set_xlabel("Time  (s)", fontsize=11)
    # ax1.set_ylabel("Distance  (m)", fontsize=11)
    # ax1.legend(loc="lower right", framealpha=0.85)
    # ax1.grid(alpha=0.25, linestyle="--")
    # fig1.tight_layout()

    # # ── Plot 2: Velocity vs Time ──────────────────────────────────────
    # fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    # fig2.patch.set_facecolor("#f8fafc")
    # ax2.set_facecolor("#f8fafc")

    # ax2.plot(
    #     t_actual, vel_fit,
    #     linewidth=2.5, color="#dc2626",
    #     label="Velocity  (dŷ/dt)"
    # )
    # ax2.axhline(
    #     vt, linestyle=":", linewidth=2.5, color="black",
    #     label=f"Terminal Velocity = {vt:.4f}  m/s"
    # )
    # ax2.set_xlabel("Time  (s)", fontsize=11)
    # ax2.set_ylabel("Velocity  (m/s)", fontsize=11)
    # ax2.legend(framealpha=0.85)
    # ax2.grid(alpha=0.25, linestyle="--")
    # fig2.tight_layout()

    # return fig1, fig2, vt

    # # ── Plot 1: Distance vs Time ──────────────────────────────────────
    # fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    # fig1.patch.set_facecolor("#f8fafc")
    # ax1.set_facecolor("#f8fafc")
 
    # step = max(1, len(t_clean) // 25)
    # ax1.scatter(
    #     t_clean[::step], y_clean[::step],
    #     s=45, color="#3b82f6", alpha=0.85, zorder=3,
    #     label="Measured Position (sampled)"
    # )
    # ax1.plot(
    #     t_actual, pos_fit,
    #     "-", linewidth=2.5, color="#dc2626", zorder=4,
    #     label=f"Polynomial Fit  (degree {poly_degree})"
    # )
    # ax1.text(
    #     0.5, 0.97,
    #     f"{eq_str}\nR² = {r2:.4f}",
    #     transform=ax1.transAxes, fontsize=8.5,
    #     verticalalignment="top", horizontalalignment="center",
    #     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
    #               edgecolor="#cccccc", alpha=0.95)
    # )
    # ax1.set_xlabel("Time  (s)", fontsize=11)
    # ax1.set_ylabel("Distance  (m)", fontsize=11)
    # ax1.legend(loc="lower right", framealpha=0.85)
    # ax1.grid(alpha=0.25, linestyle="--")
    # fig1.tight_layout()
    # fig1.subplots_adjust(left=0.15, top=0.92)
 
    # # ── Plot 2: Velocity vs Time ──────────────────────────────────────
    # fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    # fig2.patch.set_facecolor("#f8fafc")
    # ax2.set_facecolor("#f8fafc")
 
    # ax2.plot(
    #     t_actual, vel_fit,
    #     linewidth=2.5, color="#dc2626",
    #     label="Velocity  (dŷ/dt)"
    # )
    # ax2.axhline(
    #     vt, linestyle=":", linewidth=2.5, color="black",
    #     label=f"Terminal Velocity = {vt:.4f}  m/s"
    # )
    # ax2.set_xlabel("Time  (s)", fontsize=11)
    # ax2.set_ylabel("Velocity  (m/s)", fontsize=11)
    # ax2.legend(framealpha=0.85)
    # ax2.grid(alpha=0.25, linestyle="--")
    # fig2.tight_layout()
    # fig2.subplots_adjust(left=0.15, top=0.92)
 
    # return fig1, fig2, vt

    


# ═══════════════════════════════════════════════════════════════════
# FULL ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════

def analyze(video, ref_m, points_state, dp, rho_p, rho_f):
    pixel_to_m = compute_scale(points_state, ref_m)
    if pixel_to_m is None:
        raise gr.Error("Please select two calibration points first.")

    times     = []
    positions = []

    for frame_rgb, t_list, pos_list in process_video(video, pixel_to_m):
        times     = t_list
        positions = pos_list
        yield frame_rgb, None, None, None, None, None, None, None

    if len(positions) == 0:
        raise gr.Error("Particle was not detected in the video.")

    save_position_to_excel(times, positions)

    fig_dist, fig_vel, vt = create_plots(times, positions, poly_degree=POLY_DEGREE)
    vt_calc = round(vt, 4)

    Cd_exp = experimental_cd(dp, rho_p, rho_f, vt_calc)
    Re, mu_exp = compute_viscosity(rho_f, vt_calc, dp, Cd_exp)
    regime = flow_regime(Re)

    yield None, fig_dist, fig_vel, vt_calc, Cd_exp, round(Re, 4), round(mu_exp, 6), regime


# ═══════════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════════

theme = gr.themes.Soft(primary_hue="blue")

with gr.Blocks(
    theme=theme,
    title="Terminal Velocity & Viscosity Tracker",
    css="""
    /* ── Remove browser/Gradio top margin ── */
    body, html {
        margin: 0 !important;
        padding: 0 !important;
    }
    /* ── Kill Gradio's outer padding so header is truly full-width ── */
    .gradio-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
    }
    .main > .flex {
        padding: 0 !important;
        gap: 0 !important;
    }
    .gradio-container > .main > .flex > .flex:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* ── Header: flush full-width banner ── */
    .header-wrap {
        width: 100%;
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        color: white;
        padding: 20px 40px 22px 40px;
        text-align: center;
        border-radius: 0 0 20px 20px;
        margin-bottom: 20px;
        box-sizing: border-box;
    }
    .header-wrap h1 {
        color: white !important;
        margin: 0;
        font-size: 24px;
        font-weight: 700;
    }
    .header-wrap p {
        color: rgba(255,255,255,0.88) !important;
        margin-top: 6px;
        font-size: 13px;
    }

    /* ── Consistent side padding for all content ── */
    .content-area {
        padding: 0 20px 20px 20px;
    }

    /* ── Video / image boxes ── */
    #video_box video, #image_box img {
        height: 300px !important;
        width: 100% !important;
        object-fit: contain !important;
    }

    /* ── Live detection full-width ── */
    #live_detect img {
        width: 100% !important;
        max-height: 420px !important;
        object-fit: contain !important;
    }

    .card { border-radius: 12px; padding: 10px; }
    .warning { color: #b45309; font-size: 12px; margin-top: 4px; }
    """
) as demo:

    # ── Header — truly full-width, no side gaps ────────────────────────
    gr.HTML("""
    <div class="header-wrap">
        <h1>Terminal Velocity &amp; Viscosity Tracker AI</h1>
        <p>Real-time terminal velocity & viscosity estimation using AI-powered computer vision</p>
    </div>
    """)

    # ── Content with consistent padding ───────────────────────────────
    with gr.Column(elem_classes="content-area"):

        # ── Step 1 & 2 ───────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes="card"):
                    gr.Markdown("### 📱 Step 1: Upload Video")
                    
                    video_input = gr.Video(
                        label="Upload Experimental Video",
                        elem_id="video_box"
                    )

            with gr.Column():
                with gr.Group(elem_classes="card"):
                    gr.Markdown("### 📍 Step 2: Calibration")
        
                    frame_display = gr.Image(
                        label="Click any two points",
                        elem_id="image_box", interactive=True
                    )
                    ref_m = gr.Number(
                        label="Reference Distance Between Selected Points (m)",
                        value=0
                    )

        # ── Step 3: all 3 inputs on ONE row ──────────────────────────
        with gr.Group(elem_classes="card"):
            gr.Markdown("### ⚙️ Step 3: Parameters")
            with gr.Row():
                dp    = gr.Number(label="Pellet Diameter, dp (m)",     value=0)
                rho_p = gr.Number(label="Pellet Density, ρp (kg/m³)", value=0)
                rho_f = gr.Number(label="Fluid Density, ρf (kg/m³)",  value=0)
                

        # ── Run Button ────────────────────────────────────────────────
        run_btn = gr.Button("🚀 Run Analysis", variant="primary", size="lg")

        # ── Row 1: YOLO Live Detection — full width ───────────────────
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes="card"):
                    gr.Markdown("### 🎯 Live Detection")
                    live_video = gr.Image(
                        label="YOLO Tracking Output",
                        elem_id="live_detect"
                    )

        # ── Row 2: Both plots side by side ───────────────────────────
        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes="card"):
                    gr.Markdown("### 📏 Position Analysis")
                    plot_dist = gr.Plot(label="Distance vs Time (m)")

            with gr.Column():
                with gr.Group(elem_classes="card"):
                    gr.Markdown("### 📊 Velocity Analysis")
                    plot_vel = gr.Plot(label="Velocity vs Time (m/s)")

        # ── Results ───────────────────────────────────────────────────
        with gr.Group(elem_classes="card"):
            gr.Markdown("### 📈 Results")

            with gr.Row():
                vt_out     = gr.Number(label="Terminal Velocity, Vt (m/s)")
                Cd_exp_out = gr.Number(label="Experimental Drag Coefficient, Cd (-)")

            with gr.Row():
                Re_out     = gr.Number(label="Reynolds Number, Re (-)")
                mu_exp_out = gr.Number(label="Experimental Viscosity, μ (Pa·s)")

            regime_out = gr.Textbox(
                label="Flow Regime",
                interactive=False
            )

            gr.Markdown(
                "<span class='warning'>⚠️ Viscosity calculated using Stokes law: "
                "Cd = 24/Re. Valid only when Re &lt; 1 (Stokes flow). "
                "Check the Flow Regime field above.</span>"
            )

        # ── Footer ────────────────────────────────────────────────────
        gr.Markdown("""
        ---
        👩‍🔬 AI-based Fluid Mechanics Analysis System — Estimate terminal velocity & viscosity from a video
        """)

    # ── State + Events ────────────────────────────────────────────────
    points_state = gr.State({"start": None, "points": []})

    video_input.change(
        get_first_frame, inputs=video_input, outputs=frame_display
    ).then(reset_points, outputs=points_state)

    frame_display.select(
        record_click,
        inputs=[points_state, frame_display],
        outputs=[points_state, frame_display]
    )

    run_btn.click(
        analyze,
        inputs=[video_input, ref_m, points_state, dp, rho_p, rho_f],
        outputs=[live_video, plot_dist, plot_vel,
                 vt_out, Cd_exp_out, Re_out, mu_exp_out, regime_out],
        show_progress=True
    )

demo.launch(server_name="0.0.0.0")
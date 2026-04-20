import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os
from io import StringIO

# Try imports for Real Model (if available)
try:
    import joblib
    from tensorflow.keras.models import load_model
    # Added sklearn for metrics
    from sklearn.metrics import precision_recall_fscore_support 
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroMove | HAR Biomechanics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. DOMAIN LOGIC (SKELETON, MODEL, METS)
# -----------------------------------------------------------------------------

ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

# MET (Metabolic Equivalent of Task) Values for Calorie Calculation
MET_LOOKUP = {
    "WALKING": 3.8,
    "WALKING_UPSTAIRS": 8.0,
    "WALKING_DOWNSTAIRS": 3.0,
    "SITTING": 1.3,
    "STANDING": 1.5,
    "LAYING": 1.0
}

# --- SKELETON THEMES ---
THEME_STYLES = {
    # --- DEFAULT ---
    "Default": {"joint": "#10b981", "bone": "rgba(255, 255, 255, 0.6)"}, # Classic Green & White

    # --- CLASSY / MINIMALIST ---
    "Obsidian (Pro Dark)":   {"joint": "#ffffff", "bone": "#404040"},       # Bright White joints on Dark Grey bones
    "Slate & Stone (Light)": {"joint": "#1e293b", "bone": "#94a3b8"},       # Dark Slate joints on Grey bones
    "Midnight Executive":    {"joint": "#60a5fa", "bone": "#1e3a8a"},       # Bright Blue joints on Navy bones
    "Sage Green":            {"joint": "#1b5e20", "bone": "#81c784"},       # Forest Green joints on Sage bones
    "Ivory Gold":            {"joint": "#d4af37", "bone": "#b48e26"},       # Gold joints on Darker Gold bones

    # --- GIRLY POP / VIBRANT ---
    "Bubblegum Pop":         {"joint": "#ff69b4", "bone": "#a6c1ee"},       # Hot Pink joints on Pastel Blue bones
    "Lavender Haze":         {"joint": "#7c3aed", "bone": "#c4b5fd"},       # Violet joints on Soft Purple bones
    "Peach Fuzz":            {"joint": "#ea580c", "bone": "#fdba74"},       # Deep Orange joints on Peach bones
    "Electric Barbie":       {"joint": "#00ffff", "bone": "#ff00cc"},       # Cyan joints on Electric Pink bones
    "Mint Fresh":            {"joint": "#047857", "bone": "#6ee7b7"}        # Emerald joints on Mint bones
}

# --- UI THEMES (Classy & Girly Pop) ---
UI_THEMES = {
    # --- CLASSY / MINIMALIST ---
    "Obsidian (Pro Dark)": {
        "bg": "#0f0f0f",
        "text": "#e0e0e0",
        "card_bg": "rgba(255, 255, 255, 0.05)",
        "chart_text": "#e0e0e0",
        "sidebar_bg": "#000000",
        "sidebar_border": "#333333",
        "button_bg": "linear-gradient(90deg, #333333, #555555)",
        "title_gradient": "linear-gradient(to right, #ffffff, #a1a1aa)"
    },
    "Slate & Stone (Light)": {
        "bg": "#f8fafc",
        "text": "#334155",
        "card_bg": "#ffffff",
        "chart_text": "#334155",
        "sidebar_bg": "#f1f5f9",
        "sidebar_border": "#e2e8f0",
        "button_bg": "linear-gradient(90deg, #475569, #64748b)",
        "title_gradient": "linear-gradient(to right, #1e293b, #475569)"
    },
    "Midnight Executive": {
        "bg": "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
        "text": "#f8fafc",
        "card_bg": "rgba(30, 41, 59, 0.7)",
        "chart_text": "white",
        "sidebar_bg": "#111827",
        "sidebar_border": "#334155",
        "button_bg": "linear-gradient(90deg, #1d4ed8, #1e40af)",
        "title_gradient": "linear-gradient(to right, #60a5fa, #3b82f6)"
    },
    "Sage Green": {
        "bg": "linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)",
        "text": "#1b5e20",
        "card_bg": "rgba(255, 255, 255, 0.6)",
        "chart_text": "#1b5e20",
        "sidebar_bg": "#f1f8e9",
        "sidebar_border": "#a5d6a7",
        "button_bg": "linear-gradient(90deg, #66bb6a, #43a047)",
        "title_gradient": "linear-gradient(to right, #2e7d32, #66bb6a)"
    },
    "Ivory Gold": {
        "bg": "#fdfbf7",
        "text": "#4a4a4a",
        "card_bg": "#ffffff",
        "chart_text": "#4a4a4a",
        "sidebar_bg": "#fcf8f2",
        "sidebar_border": "#e5e0d8",
        "button_bg": "linear-gradient(90deg, #d4af37, #c5a028)",
        "title_gradient": "linear-gradient(to right, #b48e26, #d4af37)"
    },

    # --- GIRLY POP / VIBRANT ---
    "Bubblegum Pop": {
        "bg": "linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)",
        "text": "#4a4a4a",
        "card_bg": "rgba(255, 255, 255, 0.6)",
        "chart_text": "#2d3748",
        "sidebar_bg": "rgba(255, 255, 255, 0.8)",
        "sidebar_border": "#ffb6c1",
        "button_bg": "linear-gradient(90deg, #ff9a9e, #fad0c4)",
        "title_gradient": "linear-gradient(to right, #ff758c, #ff7eb3)"
    },
    "Lavender Haze": {
        "bg": "linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)",
        "text": "#4c1d95",
        "card_bg": "rgba(255, 255, 255, 0.5)",
        "chart_text": "#4c1d95",
        "sidebar_bg": "#f3e8ff",
        "sidebar_border": "#d8b4fe",
        "button_bg": "linear-gradient(90deg, #a78bfa, #c084fc)",
        "title_gradient": "linear-gradient(to right, #7c3aed, #a78bfa)"
    },
    "Peach Fuzz": {
        "bg": "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
        "text": "#9a3412",
        "card_bg": "rgba(255, 255, 255, 0.6)",
        "chart_text": "#9a3412",
        "sidebar_bg": "#fff7ed",
        "sidebar_border": "#fdba74",
        "button_bg": "linear-gradient(90deg, #fbbf24, #f97316)",
        "title_gradient": "linear-gradient(to right, #ea580c, #f97316)"
    },
    "Electric Barbie": {
        "bg": "linear-gradient(135deg, #ff00cc 0%, #333399 100%)",
        "text": "#ffffff",
        "card_bg": "rgba(255, 255, 255, 0.15)",
        "chart_text": "white",
        "sidebar_bg": "#500033",
        "sidebar_border": "#ff00cc",
        "button_bg": "linear-gradient(90deg, #ff1493, #db2777)",
        "title_gradient": "linear-gradient(to right, #ff69b4, #ff1493)"
    },
    "Mint Fresh": {
        "bg": "linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)",
        "text": "#064e3b",
        "card_bg": "rgba(255, 255, 255, 0.7)",
        "chart_text": "#064e3b",
        "sidebar_bg": "#ecfdf5",
        "sidebar_border": "#6ee7b7",
        "button_bg": "linear-gradient(90deg, #34d399, #10b981)",
        "title_gradient": "linear-gradient(to right, #059669, #10b981)"
    }
}

# --- Skeleton Definitions ---
JOINT_NAMES = [
    "Head", "Neck",
    "Shoulder_L", "Elbow_L", "Wrist_L",
    "Shoulder_R", "Elbow_R", "Wrist_R",
    "Spine",
    "Hip_L", "Knee_L", "Ankle_L",
    "Hip_R", "Knee_R", "Ankle_R"
]

BASE_POSE = np.array([
    [0, 1.85, 0],     # 0: Head
    [0, 1.70, 0],     # 1: Neck
    [-0.35, 1.55, 0], # 2: Shoulder_L
    [-0.5, 1.3, 0],   # 3: Elbow_L
    [-0.6, 1.05, 0],  # 4: Wrist_L
    [0.35, 1.55, 0],  # 5: Shoulder_R
    [0.5, 1.3, 0],    # 6: Elbow_R
    [0.6, 1.05, 0],   # 7: Wrist_R
    [0, 1.15, 0],     # 8: Spine
    [-0.2, 1.05, 0],  # 9: Hip_L
    [-0.2, 0.55, 0],  # 10: Knee_L
    [-0.2, 0.1, 0],   # 11: Ankle_L
    [0.2, 1.05, 0],   # 12: Hip_R
    [0.2, 0.55, 0],   # 13: Knee_R
    [0.2, 0.1, 0]     # 14: Ankle_R
])

SKELETON_EDGES = [
    (0,1),(1,8),
    (1,2),(2,3),(3,4),
    (1,5),(5,6),(6,7),
    (8,9),(9,10),(10,11),
    (8,12),(12,13),(13,14)
]

def get_activity_pose(activity: str, phase: float = 0.0) -> np.ndarray:
    """
    Return skeleton pose variation for each activity.
    phase: 0 to 2*pi for animation cycle.
    """
    pose = BASE_POSE.copy()
    
    # Simple Animation Physics (Sine Waves)
    swing = np.sin(phase)
    cos_swing = np.cos(phase)

    if activity == "WALKING":
        # Base Offset
        pose[10,2] += 0.1; pose[11,2] += 0.1
        pose[13,2] -= 0.1; pose[14,2] -= 0.1
        
        # Dynamic Animation
        # Legs (swing opposite)
        pose[10, 2] += swing * 0.3  # Left Knee
        pose[11, 2] += swing * 0.5  # Left Ankle
        pose[11, 1] += max(0, swing * 0.2) # Lift foot
        
        pose[13, 2] -= swing * 0.3  # Right Knee
        pose[14, 2] -= swing * 0.5  # Right Ankle
        pose[14, 1] += max(0, -swing * 0.2) # Lift foot
        
        # Arms (swing opposite to legs)
        pose[4, 2] -= swing * 0.3
        pose[7, 2] += swing * 0.3

    elif activity == "WALKING_UPSTAIRS":
        # High knees animation
        pose[0, 1] += 0.1
        
        # Exaggerated vertical step
        step_h = max(0, swing * 0.4)
        step_h2 = max(0, -swing * 0.4)
        
        pose[10, 1] += step_h + 0.2
        pose[11, 1] += step_h + 0.1
        pose[11, 2] += swing * 0.2
        
        pose[13, 1] += step_h2 + 0.2
        pose[14, 1] += step_h2 + 0.1
        pose[14, 2] -= swing * 0.2

    elif activity == "WALKING_DOWNSTAIRS":
        # General Posture: Leaning forward, looking down
        pose[0, 2] += 0.2 # Head forward
        pose[1, 2] += 0.1 # Neck forward
        pose[8, 2] += 0.1 # Spine forward
        
        # Arms slightly out for balance
        pose[4, 0] -= 0.1 # Left arm out
        pose[7, 0] += 0.1 # Right arm out
        
        # Step Mechanics (Reaching DOWN is key)
        # Left Leg
        pose[10, 2] += 0.1 + swing * 0.1 # Knee forward/back
        pose[11, 1] -= 0.2 + max(0, swing * 0.3) # Ankle drops significantly on swing
        pose[11, 2] += 0.2 + swing * 0.2 # Reaches forward
        
        # Right Leg
        pose[13, 2] += 0.1 - swing * 0.1
        pose[14, 1] -= 0.2 + max(0, -swing * 0.3) # Ankle drops significantly on its swing
        pose[14, 2] += 0.2 - swing * 0.2

    elif activity == "SITTING":
        pose[8, 1] -= 0.4
        pose[0:8, 1] -= 0.4
        pose[9:15, 2] += 0.5
        pose[10,1] = 0.5; pose[13,1] = 0.5
        pose[11,1] = 0.1; pose[14,1] = 0.1
        
        # Idle breathing
        pose[0, 1] += swing * 0.01 # Head bob
        pose[4, 1] += cos_swing * 0.02 # Hands

    elif activity == "STANDING":
        # Slight sway
        pose[0, 0] += swing * 0.02
        pose[4, 2] += cos_swing * 0.05
        pose[7, 2] += cos_swing * 0.05

    elif activity == "LAYING":
        pose[:, 2] = BASE_POSE[:, 1]
        pose[:, 1] = 0.2
        pose[0, 1] = 0.35
        pose[1, 1] = 0.3
        pose[3, 0] = -0.45; pose[4, 0] = -0.4
        pose[6, 0] = 0.45;  pose[7, 0] = 0.4

    return pose

class HARSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.real_mode_available = False
        
        # Try loading real files if they exist
        if HAS_ML_DEPS and os.path.exists("har_lstm_model.h5") and os.path.exists("scaler.joblib"):
            try:
                self.model = load_model("har_lstm_model.h5")
                self.scaler = joblib.load("scaler.joblib")
                self.real_mode_available = True
                print("Real Model Loaded")
            except Exception as e:
                print(f"Error loading model: {e}")
        
    def predict(self, input_data, force_sim=False):
        """
        Input: numpy array (561,)
        Output: activity_name, confidence, probabilities, is_simulated
        """
        if self.real_mode_available and not force_sim:
            try:
                scaled = self.scaler.transform(input_data.reshape(1, -1))
                reshaped = scaled.reshape(1, 1, 561)
                probs = self.model.predict(reshaped)[0]
                idx = np.argmax(probs)
                return ACTIVITIES[idx], float(probs[idx]), probs, False
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                return "ERROR", 0.0, np.zeros(6), False

        else:
            time.sleep(0.1) # Fast simulate
            data_hash = hash(input_data.tobytes()) % (2**32 - 1)
            rs = np.random.RandomState(data_hash)
            probs = rs.dirichlet(np.ones(6) * 0.5) 
            max_idx = rs.randint(0, 6)
            probs[max_idx] += 20.0 # High confidence
            probs = probs / probs.sum()
            idx = np.argmax(probs)
            return ACTIVITIES[idx], float(probs[idx]), probs, True

    def explain_features(self, input_data):
        """Simple heuristic to explain which sensors are active."""
        acc_energy = np.mean(np.abs(input_data[:265]))
        gyro_energy = np.mean(np.abs(input_data[266:]))
        
        total = acc_energy + gyro_energy + 1e-9
        return {
            "Accelerometer": acc_energy / total,
            "Gyroscope": gyro_energy / total
        }

# -----------------------------------------------------------------------------
# 3. PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

def build_skeleton_figure(activity: str, confidence: float, theme="Default", animate=False) -> go.Figure:
    # Get Theme Colors
    styles = THEME_STYLES.get(theme, THEME_STYLES["Default"])
    
    # Override joint color if confidence is low, otherwise use theme
    if confidence < 0.6:
        joint_color = "#f59e0b" # Warning amber
    else:
        joint_color = styles["joint"]
        
    bone_color = styles["bone"]
    
    # Helper to build a single frame trace set
    def get_traces(phase=0.0):
        pose = get_activity_pose(activity, phase)
        x, z, y = pose[:, 0], pose[:, 1], pose[:, 2] # Map: Y=Depth, Z=Height
        
        # Optimize Bone Lines: Create a single trace with None separators
        x_bones, y_bones, z_bones = [], [], []
        for i, j in SKELETON_EDGES:
            x_bones.extend([x[i], x[j], None])
            y_bones.extend([y[i], y[j], None])
            z_bones.extend([z[i], z[j], None])
            
        return [
            go.Scatter3d(
                x=x_bones, y=y_bones, z=z_bones,
                mode="lines",
                line=dict(color=bone_color, width=6),
                hoverinfo='none',
                name='Bones'
            ),
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(size=10, color=joint_color, line=dict(color='white', width=2), opacity=0.9),
                text=JOINT_NAMES,
                hoverinfo='text',
                name='Joints'
            )
        ]

    # Initial Data
    initial_traces = get_traces(phase=0.0)
    
    fig = go.Figure(
        data=initial_traces,
        layout=go.Layout(
            title=dict(text=f"Biomechanics: {activity}", font=dict(color='gray' if 'Light' in theme else 'white')),
            scene=dict(
                xaxis=dict(visible=False, showgrid=False, range=[-1, 1]),
                yaxis=dict(visible=False, showgrid=False, range=[-1, 2.5]),
                zaxis=dict(visible=False, showgrid=False, range=[0, 2]),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5), up=dict(x=0, y=0, z=1))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
            showlegend=False
        )
    )

    # Add Animation Frames if requested
    if animate:
        frames = []
        # Create 20 frames for a full cycle loop
        for t in np.linspace(0, 6.28, 20):
            frame_traces = get_traces(phase=t)
            frames.append(go.Frame(data=frame_traces, traces=[0, 1]))
            
        fig.frames = frames
        
        # Add Play/Pause Buttons
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=0.1,
                x=0.1,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0))]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
                )]
            )]
        )

    return fig

def build_waveform_figure(sample: np.ndarray, split=False, text_color="white") -> go.Figure:
    fig = go.Figure()
    
    if split:
        fig.add_trace(go.Scatter(y=sample[:100], name="Body Acc", line=dict(color='#3b82f6')))
        fig.add_trace(go.Scatter(y=sample[266:366], name="Gyroscope", line=dict(color='#ef4444')))
        fig.update_layout(title="Split Sensor Analysis")
    else:
        fig.add_trace(go.Scatter(
            y=sample,
            mode='lines',
            line=dict(color='#3b82f6', width=1),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',
            name="Raw Feature Vector"
        ))
        fig.update_layout(title="Full Feature Waveform (561 pts)")
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, color='#94a3b8'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=False, color='#94a3b8'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        legend=dict(font=dict(color=text_color)),
        title=dict(font=dict(color=text_color))
    )
    return fig

def build_gauge(conf: float, text_color="white") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        number={'suffix': "%", 'font': {'color': text_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': text_color},
            'bar': {'color': "#10b981"},
            'bgcolor': "rgba(128,128,128,0.2)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.3)"},
                {'range': [50, 80], 'color': "rgba(245, 158, 11, 0.3)"}
            ],
            'threshold': {'line': {'color': text_color, 'width': 2}, 'thickness': 0.75, 'value': 90}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': text_color, 'family': "Arial"},
        margin=dict(l=20, r=20, t=30, b=20),
        height=200
    )
    return fig

def generate_report(activity, confidence, energy, calories=None):
    """Generate a simple text report."""
    cal_str = f"{calories:.2f} kcal/min" if calories is not None else "Disabled"
    
    report = f"""
    NEUROMOVE HAR - ANALYSIS REPORT
    ==============================
    Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
    
    PREDICTION RESULTS
    ------------------
    Detected Activity: {activity}
    Confidence Score:  {confidence*100:.2f}%
    Signal Intensity:  {energy:.2f} J
    Est. Burn Rate:    {cal_str}
    
    SYSTEM DIAGNOSTICS
    ------------------
    Model Version:      v2.1 Stable
    Sensor Status:      Online
    
    RECOMMENDATION
    --------------
    Activity classification is consistent with biomechanical patterns.
    """
    return report

# -----------------------------------------------------------------------------
# 4. LAYOUT RENDERERS (NEW!)
# -----------------------------------------------------------------------------

def render_layout(layout_name, activity, confidence, energy, calories_per_min, sample_data, selected_theme, current_theme, animate=False, actual_label=None, explanation=None, enable_download=True):
    """
    Handles drawing the UI based on the selected layout mode.
    """
    # Define text colors for charts
    chart_text = current_theme['chart_text']
    
    # --- LAYOUT 1: STANDARD DASHBOARD (GRID) ---
    if layout_name == "Standard Grid":
        # Row 1: KPIs
        if calories_per_min is not None:
            k1, k2, k3, k4 = st.columns(4)
        else:
            k1, k2, k3 = st.columns(3)

        with k1:
             st.markdown(f'<div class="glass-card hover-lift"><p class="sub-metric">Activity</p><h3 style="color: #3b82f6; margin:0;">{activity}</h3></div>', unsafe_allow_html=True)
        with k2:
             st.markdown(f'<div class="glass-card hover-lift"><p class="sub-metric">Confidence</p><h3 style="color: #10b981; margin:0;">{confidence*100:.1f}%</h3></div>', unsafe_allow_html=True)
        with k3:
             st.markdown(f'<div class="glass-card hover-lift"><p class="sub-metric">Intensity</p><h3 style="color: {current_theme["text"]}; margin:0;">{energy:.2f} J</h3></div>', unsafe_allow_html=True)
        
        if calories_per_min is not None:
            with k4:
                st.markdown(f'<div class="glass-card hover-lift"><p class="sub-metric">Est. Burn</p><h3 style="color: #f59e0b; margin:0;">{calories_per_min:.1f} <span style="font-size:0.8rem">kcal/min</span></h3></div>', unsafe_allow_html=True)

        # Row 2: Visuals
        c_vis1, c_vis2 = st.columns([1, 1])
        with c_vis1:
            st.subheader("3D Biomechanics")
            st.plotly_chart(build_skeleton_figure(activity, confidence, theme=selected_theme, animate=animate), use_container_width=True, key=f"std_skel_{time.time()}")
            if explanation:
                st.subheader("Sensor Contribution")
                df_exp = pd.DataFrame(list(explanation.items()), columns=["Sensor", "Contribution"])
                fig_exp = px.bar(df_exp, x="Contribution", y="Sensor", orientation='h', color="Sensor", color_discrete_sequence=['#3b82f6', '#ef4444'])
                fig_exp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=chart_text), height=150, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_exp, use_container_width=True)

        with c_vis2:
            st.subheader("Signal Waveform")
            st.plotly_chart(build_waveform_figure(sample_data, split=True, text_color=chart_text), use_container_width=True, key=f"std_wave_{time.time()}")
            
            st.markdown("---")
            c_gauge, c_dl = st.columns([2, 1])
            with c_gauge:
                st.plotly_chart(build_gauge(confidence, text_color=chart_text), use_container_width=True, key=f"std_gauge_{time.time()}")
            
            if enable_download:
                with c_dl:
                    st.write("") # Spacer
                    st.write("") 
                    report_txt = generate_report(activity, confidence, energy, calories_per_min)
                    st.download_button(label="Report", data=report_txt, file_name="report.txt", mime="text/plain", use_container_width=True)

    # --- LAYOUT 2: SPLIT FOCUS (COMMAND CENTER) ---
    elif layout_name == "Split Focus":
        col_main, col_sidebar = st.columns([1.5, 1])
        
        with col_main:
            # Huge 3D View
            st.markdown("### Movement Analysis")
            st.plotly_chart(build_skeleton_figure(activity, confidence, theme=selected_theme, animate=animate), use_container_width=True, key=f"split_skel_{time.time()}")
            st.plotly_chart(build_waveform_figure(sample_data, split=False, text_color=chart_text), use_container_width=True, key=f"split_wave_{time.time()}")

        with col_sidebar:
            st.markdown("### Metrics")
            st.markdown(f'<div class="glass-card hover-lift"><h2 style="color: #3b82f6; margin:0;">{activity}</h2><p class="sub-metric">Detected Activity</p></div>', unsafe_allow_html=True)
            
            if calories_per_min is not None:
                c1, c2 = st.columns(2)
                c1.markdown(f'<div class="glass-card hover-lift" style="text-align:center;"><h3 style="color: #10b981; margin:0;">{confidence*100:.0f}%</h3><small>Conf.</small></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="glass-card hover-lift" style="text-align:center;"><h3 style="color: #f59e0b; margin:0;">{calories_per_min:.1f}</h3><small>kcal</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="glass-card hover-lift" style="text-align:center;"><h3 style="color: #10b981; margin:0;">{confidence*100:.0f}%</h3><small>Confidence</small></div>', unsafe_allow_html=True)
            
            st.plotly_chart(build_gauge(confidence, text_color=chart_text), use_container_width=True, key=f"split_gauge_{time.time()}")
            
            if actual_label:
                if activity == actual_label:
                    st.success("MATCH: Ground Truth Verified")
                else:
                    st.error(f"MISMATCH: Expected {actual_label}")

    # --- LAYOUT 3: VERTICAL STACK (MOBILE) ---
    elif layout_name == "Vertical Stack":
        # Compact Header
        st.markdown(f"""
        <div style="background: rgba(59, 130, 246, 0.2); padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h2 style="margin:0; color: #3b82f6;">{activity}</h2>
            <p style="margin:0; opacity: 0.8;">Confidence: {confidence*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(build_skeleton_figure(activity, confidence, theme=selected_theme, animate=animate), use_container_width=True, key=f"vert_skel_{time.time()}")
        
        with st.expander("Signal Data", expanded=True):
            st.plotly_chart(build_waveform_figure(sample_data, split=True, text_color=chart_text), use_container_width=True, key=f"vert_wave_{time.time()}")
            
        with st.expander("Detailed Metrics"):
            st.write(f"**Energy Intensity:** {energy:.2f} J")
            if calories_per_min is not None:
                st.write(f"**Calories:** {calories_per_min:.2f} kcal/min")
            if actual_label:
                st.write(f"**Ground Truth:** {actual_label}")

# -----------------------------------------------------------------------------
# 5. MAIN APPLICATION
# -----------------------------------------------------------------------------

def main():
    system = HARSystem()
    
    # --- Sidebar Controls ---
    with st.sidebar:
        st.markdown("### Control Panel")
        mode = st.selectbox("Operation Mode", ["Single Sample", "Live Stream (Sim)", "Batch Analysis"])
        
        st.markdown("---")
        st.markdown("### User Profile")
        
        enable_calories = st.checkbox("Enable Calorie Estimation", value=False)
        if enable_calories:
            user_weight = st.number_input("Body Weight (kg)", min_value=30, max_value=200, value=75, step=1)
        else:
            user_weight = None
        
        st.markdown("---")
        st.markdown("### Visual Settings")
        
        # LAYOUT SELECTOR (NEW)
        layout_mode = st.selectbox("Layout Mode", ["Standard Grid", "Split Focus", "Vertical Stack"])
        
        # UI THEME SELECTOR
        ui_theme_name = st.selectbox("Interface Theme", list(UI_THEMES.keys()))
        
        # SKELETON THEME SELECTOR (SEPARATE)
        selected_theme = st.selectbox("Skeleton Style", list(THEME_STYLES.keys()))
        
        # ANIMATION TOGGLE (NEW!)
        # Only show animation toggle in Single Sample mode
        if mode == "Single Sample":
            animate_skeleton = st.checkbox("Animate Movement", value=True)
        else:
            animate_skeleton = False
        
        st.markdown("---")
        # System Status
        st.markdown("### System Status")
        # Add pulse animation to status
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="width: 10px; height: 10px; background-color: #4ade80; border-radius: 50%; margin-right: 10px; animation: pulse 2s infinite;"></div>
            <span>Online</span>
        </div>
        <style>
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(74, 222, 128, 0); }
                100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
            }
        </style>
        """, unsafe_allow_html=True)

        if system.real_mode_available:
            st.success("Model Loaded")
        else:
            st.warning("Simulation Mode")
            
        st.markdown("*Debug Info*")
        st.checkbox("Show Raw Probabilities", value=False, key="show_probs")

    # --- THEME APPLICATION ---
    current_theme = UI_THEMES[ui_theme_name]
    
    # Inject CSS dynamically based on selection
    st.markdown(f"""
    <style>
        /* Global Theme & Animated Background */
        .stApp {{
            background: {current_theme['bg']};
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: {current_theme['text']};
        }}
        
        @keyframes gradientBG {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {current_theme['sidebar_bg']}; 
            /* Gradient for sidebar to make it look nicer */
            background-image: linear-gradient(180deg, {current_theme['sidebar_bg']} 0%, rgba(0,0,0,0.2) 100%);
            border-right: 1px solid {current_theme['sidebar_border']};
        }}
        
        /* Sidebar Text Color Fix */
        .css-17lntkn, .css-1v0mbdj, .css-10trblm {{
            color: {current_theme['text']} !important;
        }}
        
        /* Enhanced Glassmorphism Cards with Entrance Animation */
        .glass-card {{
            background-color: {current_theme['card_bg']};
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(12px);
            margin-bottom: 20px;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: fadeInUp 0.8s ease-out backwards;
        }}
        
        /* Staggered Animation Delays for cards */
        .glass-card:nth-child(1) {{ animation-delay: 0.1s; }}
        .glass-card:nth-child(2) {{ animation-delay: 0.2s; }}
        .glass-card:nth-child(3) {{ animation-delay: 0.3s; }}
        .glass-card:nth-child(4) {{ animation-delay: 0.4s; }}

        @keyframes fadeInUp {{
            0% {{ opacity: 0; transform: translateY(20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Dramatic Hover Effect */
        .hover-lift:hover {{
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            border: 1px solid rgba(255, 255, 255, 0.3);
            z-index: 10;
        }}
        
        /* Typography */
        h1, h2, h3, p {{
            color: {current_theme['text']} !important;
        }}
        
        .sub-metric {{
            color: {current_theme['text']} !important;
            opacity: 0.7;
        }}
        
        /* Pulsing Button Styling - DYNAMIC */
        div.stButton > button {{
            background: {current_theme['button_bg']};
            background-size: 200% 200%;
            animation: gradientBG 3s ease infinite;
            border: none;
            color: white;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        div.stButton > button:hover {{
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
        }}
        
        /* Animated Text Header - DYNAMIC */
        @keyframes textShine {{
            0% {{ background-position: 0% 50%; }}
            100% {{ background-position: 100% 50%; }}
        }}
        .animated-header {{
            background: {current_theme['title_gradient']};
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: textShine 3s linear infinite;
            font-weight: 800;
        }}
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
        <div>
            <h1 style="margin-bottom: 0;" class="animated-header">NeuroMove <span style="font-weight:300;">HAR</span></h1>
            <p style="color: {current_theme['text']}; opacity: 0.8;">Advanced Human Activity Recognition Dashboard</p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; border: 1px solid rgba(16, 185, 129, 0.4);">v2.1 Stable</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODE 1: SINGLE SAMPLE ANALYSIS
    # -------------------------------------------------------------------------
    if mode == "Single Sample":
        if 'mock_data' not in st.session_state:
            st.session_state.mock_data = np.random.rand(561)

        input_method = st.radio("Input Source", ["Upload CSV", "Random Sample"], horizontal=True)
        sample_data = None
        actual_label = None # Store label if exists
        should_use_real_model = True
        
        if input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload Sensor Data (561 or 562 columns)", type=['csv'])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file, header=None)
                    num_cols = df.shape[1]
                    
                    if num_cols == 561:
                        sample_data = df.iloc[0].values
                        st.success("Data successfully loaded (Unlabeled).")
                    elif num_cols == 562:
                        # 561 Features + 1 Label
                        sample_data = df.iloc[0, :-1].values.astype(float)
                        actual_label = df.iloc[0, -1] # Last column is label
                        st.success(f"Data successfully loaded (Labeled). Actual: {actual_label}")
                    else:
                        st.error(f"Error: Expected 561 or 562 columns, found {num_cols}")
                except Exception as e:
                    st.error(f"File Error: {e}")
        else:
            should_use_real_model = False
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("Generating synthetic sensor data for demonstration.")
            with col2:
                if st.button("Generate New"):
                    st.session_state.mock_data = np.random.rand(561)
                    st.rerun()
            sample_data = st.session_state.mock_data

        if sample_data is not None:
            # Predict
            with st.spinner("Processing neural network..."):
                activity, confidence, probs, is_sim = system.predict(sample_data, force_sim=(not should_use_real_model))
                explanation = system.explain_features(sample_data)
                energy = np.mean(np.abs(sample_data)) * 10
                
                # Calorie Calculation
                calories_per_min = None
                if enable_calories and user_weight:
                    met_value = MET_LOOKUP.get(activity, 1.0)
                    calories_per_min = (met_value * 3.5 * user_weight) / 200

            # --- Validation Alert (If Labeled) ---
            if actual_label:
                if activity == actual_label:
                    st.success(f"Correct Prediction! Model detected **{activity}**, matching the ground truth.")
                else:
                    st.error(f"Mismatch! Model detected **{activity}**, but Actual Label is **{actual_label}**.")

            # --- RENDER SELECTED LAYOUT ---
            render_layout(
                layout_mode, activity, confidence, energy, calories_per_min, 
                sample_data, selected_theme, current_theme, animate_skeleton, actual_label, explanation,
                enable_download=True
            )

    # -------------------------------------------------------------------------
    # MODE 2: LIVE STREAM (SIMULATION)
    # -------------------------------------------------------------------------
    elif mode == "Live Stream (Sim)":
        st.info("Simulating real-time sensor data stream from Bluetooth device...")
        
        # 1. Initialize Session State for Data Storage
        if 'stream_history' not in st.session_state:
            st.session_state.stream_history = []

        col_start, col_stop = st.columns(2)
        start_btn = col_start.button("Start Stream", type="primary")
        stop_btn = col_stop.button("Stop Stream")
        
        # 2. Button Logic
        if start_btn:
            st.session_state.streaming = True
            st.session_state.stream_history = []  # Clear old data on new start
        
        if stop_btn:
            st.session_state.streaming = False
            
        # Placeholders for live updates
        placeholder = st.empty()
        
        # 3. Streaming Loop
        if st.session_state.get('streaming', False):
            # Simulation Loop
            for i in range(100):
                if not st.session_state.streaming:
                    break
                    
                # Generate random data
                live_data = np.random.rand(561)
                live_act, live_conf, _, _ = system.predict(live_data, force_sim=True)
                
                # Calcs
                l_cal = None
                if enable_calories and user_weight:
                    l_met = MET_LOOKUP.get(live_act, 1.0)
                    l_cal = (l_met * 3.5 * user_weight) / 200
                
                # --- SAVE DATA TO HISTORY ---
                st.session_state.stream_history.append({
                    "Packet_ID": i + 1000,
                    "Timestamp": time.strftime("%H:%M:%S"),
                    "Activity": live_act,
                    "Confidence": live_conf,
                    "Calories_Burned": l_cal
                })
                # ----------------------------

                # RENDER DYNAMICALLY INSIDE PLACEHOLDER
                with placeholder.container():
                    render_layout(
                        layout_mode, live_act, live_conf, 0.0, l_cal, 
                        live_data, selected_theme, current_theme, animate=False, enable_download=False
                    )
                
                time.sleep(1.5) # Update delay

        # 4. Export Logic (Appears when stream stops or finishes)
        if len(st.session_state.stream_history) > 0:
            st.markdown("---")
            st.success(f"Session Complete: {len(st.session_state.stream_history)} data points captured.")
            
            # Convert to DataFrame
            df_export = pd.DataFrame(st.session_state.stream_history)
            
            # CSV Conversion
            csv = df_export.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Export Session Data to CSV",
                data=csv,
                file_name=f"neuromove_session_{int(time.time())}.csv",
                mime="text/csv",
            )

    # -------------------------------------------------------------------------
    # MODE 3: BATCH ANALYSIS (UPDATED)
    # -------------------------------------------------------------------------
    elif mode == "Batch Analysis":
        st.markdown("### Batch CSV Processing")
        st.markdown("Upload a CSV file. If it has **562 columns**, the last column is treated as the **Actual Label**.")
        batch_file = st.file_uploader("Upload CSV", type=['csv'])
        
        # Add speed control
        speed = st.slider("Playback Speed (seconds per frame)", 0.05, 1.0, 0.1)

        if batch_file:
            # Read without header to handle raw numbers
            df_batch = pd.read_csv(batch_file, header=None)
            
            # --- Detect if we have labels ---
            num_cols = df_batch.shape[1]
            has_labels = False
            
            if num_cols == 562:
                has_labels = True
                st.success(f"Labeled Dataset Detected ({len(df_batch)} rows). Validation Mode Active.")
            elif num_cols == 561:
                st.info(f"Unlabeled Dataset Detected ({len(df_batch)} rows). Prediction Mode Only.")
            else:
                st.error(f"Invalid format. Expected 561 or 562 columns, found {num_cols}.")
                st.stop()
            
            if st.button("Run Analysis"):
                results = []
                correct_count = 0
                
                # Layout for live processing
                col_status, col_viz = st.columns([1, 1])
                
                with col_status:
                    st.write("Processing Status")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    metric_placeholder = st.empty()
                    
                with col_viz:
                    st.write("Live Skeleton Preview")
                    skel_placeholder = st.empty()
                
                # Limit to first 100 for demo speed, or remove [:100] for full run
                # Using a smaller slice for the loop to prevent browser freeze on large files
                processing_limit = 100 
                if len(df_batch) > processing_limit:
                    st.warning(f"Previewing first {processing_limit} rows for performance...")
                    loop_range = range(processing_limit)
                else:
                    loop_range = range(len(df_batch))

                for i in loop_range:
                    # Parse Row
                    if has_labels:
                        # Features are 0 to 560, Label is 561
                        row_features = df_batch.iloc[i, :-1].values.astype(float)
                        actual_label = df_batch.iloc[i, -1]
                    else:
                        row_features = df_batch.iloc[i].values.astype(float)
                        actual_label = "Unknown"

                    # Predict
                    act, conf, _, _ = system.predict(row_features, force_sim=not system.real_mode_available)
                    
                    # Validation Logic
                    is_correct = (act == actual_label) if has_labels else None
                    if is_correct: correct_count += 1
                    
                    status_icon = "[Match]" if is_correct else "[Mismatch]" if has_labels else "[?]"
                    
                    # Calc Calories
                    cal = None
                    if enable_calories and user_weight:
                        met = MET_LOOKUP.get(act, 1.0)
                        cal = (met * 3.5 * user_weight) / 200
                    
                    results.append({
                        "Time Step": i, 
                        "Actual": actual_label,
                        "Predicted": act, 
                        "Confidence": conf,
                        "Correct": is_correct
                    })
                    
                    # Update Visuals
                    if has_labels:
                        status_text.markdown(f"**Step {i+1}**: Actual: **{actual_label}** | Pred: **{act}** {status_icon}")
                        current_acc = (correct_count / (i+1)) * 100
                        metric_placeholder.metric("Running Accuracy", f"{current_acc:.1f}%")
                    else:
                        status_text.markdown(f"**Step {i+1}**: Pred: **{act}**")

                    skel_placeholder.plotly_chart(
                        build_skeleton_figure(act, conf, theme=selected_theme, animate=animate_skeleton), 
                        use_container_width=True,
                        key=f"batch_frame_{i}"
                    )
                    
                    progress_bar.progress((i + 1) / len(loop_range))
                    time.sleep(speed)
                
                df_results = pd.DataFrame(results)
                
                # --- Post-Analysis Stats ---
                st.markdown("---")
                st.subheader("Analysis Results")
                
                b1, b2 = st.columns(2)
                
                with b1:
                    # Confusion Matrix (Only if labeled)
                    if has_labels:
                        st.markdown("**Confusion Matrix**")
                        try:
                            # Simple cross-tabulation
                            confusion = pd.crosstab(df_results['Actual'], df_results['Predicted'])
                            fig_conf = px.imshow(confusion, text_auto=True, aspect="auto", color_continuous_scale='Blues')
                            fig_conf.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)', 
                                plot_bgcolor='rgba(0,0,0,0)', 
                                font=dict(color=current_theme['chart_text'])
                            )
                            st.plotly_chart(fig_conf, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not generate matrix: {e}")
                    else:
                        st.markdown("**Activity Distribution**")
                        fig_pie = px.pie(df_results, names="Predicted", title="Predicted Activity Breakdown", hole=0.4)
                        fig_pie.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color=current_theme['chart_text'])
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                with b2:
                    if has_labels:
                        final_acc = (df_results['Correct'].sum() / len(df_results)) * 100
                        st.markdown("**Final Accuracy Score**")
                        fig_acc = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = final_acc,
                            title = {'text': "Accuracy"},
                            gauge = {'axis': {'range': [0, 100], 'tickcolor': current_theme['chart_text']}, 'bar': {'color': "#3b82f6"}}
                        ))
                        fig_acc.update_layout(
                            height=300, 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color=current_theme['chart_text'])
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)
                    else:
                        st.markdown("**Confidence Timeline**")
                        fig_line = px.area(df_results, x="Time Step", y="Confidence", color="Predicted", title="Model Confidence")
                        fig_line.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color=current_theme['chart_text'])
                        )
                        st.plotly_chart(fig_line, use_container_width=True)

                if has_labels and HAS_ML_DEPS:
                    st.markdown("---")
                    st.subheader("Detailed Performance Metrics")
                    
                    y_true = df_results['Actual']
                    y_pred = df_results['Predicted']
                    
                    try:
                        # 1. Classification Report Table
                        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=ACTIVITIES, zero_division=0)
                        
                        metrics_df = pd.DataFrame({
                            'Activity': ACTIVITIES,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1
                        }).set_index('Activity')
                        
                        # 2. Grouped Bar Chart
                        metrics_melted = metrics_df.reset_index().melt(id_vars='Activity', var_name='Metric', value_name='Score')
                        fig_metrics = px.bar(metrics_melted, x='Activity', y='Score', color='Metric', barmode='group', title="Precision & Recall per Activity")
                        fig_metrics.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color=current_theme['chart_text'])
                        )
                        
                        # Layout
                        m1, m2 = st.columns([1, 1.5])
                        with m1:
                            st.dataframe(metrics_df.style.highlight_max(axis=0))
                            
                            # Overall Macro F1
                            macro_f1 = f1.mean()
                            st.metric("Macro F1-Score", f"{macro_f1:.2f}")
                            
                        with m2:
                            st.plotly_chart(fig_metrics, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error calculating metrics: {e}")
                    
                st.dataframe(df_results)


if __name__ == "__main__":
    main()
# ==============================================================================
# AIR QUALITY FORECASTING & EARLY WARNING SYSTEM - GRADIO APPLICATION
# ==============================================================================

import os
import gradio as gr
import pandas as pd
from predictor_engine import AQIPredictor, get_aqi_category_info

# Initialize predictor
predictor = AQIPredictor()

# Preset profiles for realistic testing
PRESETS = {
    "🌿 Clean Coastal / Mountain Air (Good)": {
        "aqi": 38, "prev_aqi": 42, "pm25": 14.0, "no2": 8.5, "co": 0.4,
        "pm10": 32.0, "no": 4.2, "nox": 11.0, "so2": 5.1, "o3": 22.0,
        "month": 7, "day_of_week": 2
    },
    "🌤️ Typical Bengaluru Spring (Satisfactory)": {
        "aqi": 76, "prev_aqi": 82, "pm25": 32.0, "no2": 21.0, "co": 0.75,
        "pm10": 68.0, "no": 10.5, "nox": 24.0, "so2": 8.4, "o3": 31.0,
        "month": 3, "day_of_week": 4
    },
    "⚠️ Moderate Urban Smog (Early Warning Triggered)": {
        "aqi": 138, "prev_aqi": 125, "pm25": 65.0, "no2": 42.0, "co": 1.3,
        "pm10": 115.0, "no": 18.0, "nox": 38.0, "so2": 14.5, "o3": 44.0,
        "month": 10, "day_of_week": 1
    },
    "🚨 High Pollution Episode (Poor)": {
        "aqi": 245, "prev_aqi": 210, "pm25": 118.0, "no2": 62.0, "co": 2.1,
        "pm10": 210.0, "no": 32.0, "nox": 65.0, "so2": 22.0, "o3": 58.0,
        "month": 11, "day_of_week": 3
    },
    "🛑 Severe Winter Inversion Crisis (Severe)": {
        "aqi": 412, "prev_aqi": 385, "pm25": 285.0, "no2": 98.0, "co": 3.9,
        "pm10": 390.0, "no": 55.0, "nox": 115.0, "so2": 34.0, "o3": 72.0,
        "month": 12, "day_of_week": 5
    }
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

CUSTOM_CSS = """
/* App styling */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #047857 100%);
    padding: 24px 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
}

.main-header h1 {
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
    color: #ffffff !important;
}

.main-header p {
    font-size: 14px;
    color: #cbd5e1 !important;
    margin: 0;
    line-height: 1.5;
}

.header-badges {
    display: flex;
    gap: 10px;
    margin-top: 12px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 500;
    color: #f1f5f9;
}

/* Prediction Cards */
.result-card {
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    transition: all 0.2s ease;
}

.alert-banner-unhealthy {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
    border: 2px solid #ef4444;
    color: #ffffff;
    padding: 16px 20px;
    border-radius: 14px;
    margin-bottom: 18px;
    box-shadow: 0 8px 20px rgba(239, 68, 68, 0.25);
}

.alert-banner-healthy {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 2px solid #10b981;
    color: #ffffff;
    padding: 16px 20px;
    border-radius: 14px;
    margin-bottom: 18px;
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.25);
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
}

.metric-pill {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}

.metric-pill .label {
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 4px;
}

.metric-pill .value {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}

.metric-pill .sub {
    font-size: 11px;
    margin-top: 4px;
    font-weight: 500;
}

.advisory-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.advisory-box h4 {
    margin: 0 0 10px 0;
    font-size: 15px;
    font-weight: 700;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 8px;
}

.advisory-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #334155;
    line-height: 1.45;
}

.advisory-item strong {
    color: #0f172a;
}
"""


def apply_preset(preset_name):
    if preset_name in PRESETS:
        p = PRESETS[preset_name]
        return (
            p["aqi"], p["prev_aqi"], p["pm25"], p["no2"], p["co"],
            p["pm10"], p["no"], p["nox"], p["so2"], p["o3"],
            MONTH_NAMES[p["month"] - 1], DAY_NAMES[p["day_of_week"]]
        )
    return (100, 100, 50.0, 25.0, 1.0, 104.2, 13.5, 27.8, 11.2, 34.6, "June", "Thursday")


def run_forecast(aqi, prev_aqi, pm25, no2, co, pm10, no, nox, so2, o3, month_str, day_str):
    try:
        month_int = MONTH_NAMES.index(month_str) + 1 if month_str in MONTH_NAMES else 6
        day_int = DAY_NAMES.index(day_str) if day_str in DAY_NAMES else 3

        res = predictor.predict(
            aqi=aqi,
            prev_aqi=prev_aqi,
            pm25=pm25,
            no2=no2,
            co=co,
            pm10=pm10,
            no=no,
            nox=nox,
            so2=so2,
            o3=o3,
            month=month_int,
            day_of_week=day_int
        )

        pred_aqi = res["predicted_aqi"]
        is_unhealthy = res["is_unhealthy"]
        prob = res["unhealthy_prob_pct"]
        cat = res["category"]
        color = res["category_color"]
        delta = res["aqi_delta"]
        impact = res["impact"]
        advisory = res["advisory"]
        baseline_flag = res["baseline_rule_flag"]

        # Alert Banner HTML
        if is_unhealthy:
            banner_html = f"""
            <div class="alert-banner-unhealthy">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:11px; text-transform:uppercase; letter-spacing:1px; font-weight:700; opacity:0.9;">
                            🚨 24-Hour Early Warning Triggered
                        </div>
                        <div style="font-size:22px; font-weight:800; margin-top:2px;">
                            UNHEALTHY AIR EXPECTED TOMORROW (AQI &gt; 100)
                        </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.2); padding:8px 14px; border-radius:10px; text-align:right;">
                        <div style="font-size:11px; opacity:0.9;">Risk Confidence</div>
                        <div style="font-size:20px; font-weight:800;">{prob}%</div>
                    </div>
                </div>
                <div style="margin-top:8px; font-size:13px; opacity:0.95;">
                    The machine learning model predicts tomorrow's AQI will breach the healthy threshold. Advance precautions are advised today.
                </div>
            </div>
            """
        else:
            banner_html = f"""
            <div class="alert-banner-healthy">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:11px; text-transform:uppercase; letter-spacing:1px; font-weight:700; opacity:0.9;">
                            ✅ Early Warning Cleared
                        </div>
                        <div style="font-size:22px; font-weight:800; margin-top:2px;">
                            HEALTHY AIR QUALITY PREDICTED (AQI &le; 100)
                        </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.2); padding:8px 14px; border-radius:10px; text-align:right;">
                        <div style="font-size:11px; opacity:0.9;">Safety Confidence</div>
                        <div style="font-size:20px; font-weight:800;">{round(100 - prob, 1)}%</div>
                    </div>
                </div>
                <div style="margin-top:8px; font-size:13px; opacity:0.95;">
                    Air quality is expected to remain within clean or satisfactory levels tomorrow. Normal outdoor routines can continue.
                </div>
            </div>
            """

        # Delta indicator text and color
        if delta > 0:
            delta_str = f"↗ +{delta} (Worsening)"
            delta_color = "#ef4444"
        elif delta < 0:
            delta_str = f"↘ {delta} (Improving)"
            delta_color = "#10b981"
        else:
            delta_str = "→ 0.0 (Unchanged)"
            delta_color = "#64748b"

        # Metric Grid HTML
        metrics_html = f"""
        <div class="metric-grid">
            <div class="metric-pill" style="border-top: 4px solid {color};">
                <div class="label">Forecasted AQI</div>
                <div class="value" style="color: {color};">{round(pred_aqi)}</div>
                <div class="sub" style="color: {color}; font-weight:700;">{cat}</div>
            </div>
            <div class="metric-pill">
                <div class="label">24h AQI Trend</div>
                <div class="value" style="font-size: 20px; color: {delta_color}; margin-top: 4px;">{delta_str}</div>
                <div class="sub" style="color:#64748b;">Today: {round(aqi)}</div>
            </div>
            <div class="metric-pill">
                <div class="label">Unhealthy Probability</div>
                <div class="value">{prob}%</div>
                <div class="sub" style="color: {'#ef4444' if prob >= 50 else '#10b981'}; font-weight:600;">
                    { 'High Risk' if prob >= 70 else ('Moderate Risk' if prob >= 50 else 'Low Risk') }
                </div>
            </div>
            <div class="metric-pill">
                <div class="label">CPCB Classification</div>
                <div class="value" style="font-size: 18px; color: {color}; margin-top: 6px;">{cat}</div>
                <div class="sub" style="color:#64748b;">Range: {res['aqi_range']}</div>
            </div>
        </div>
        """

        # Actionable Health Advisories HTML
        advisory_html = f"""
        <div class="advisory-box">
            <h4>🛡️ Actionable Health Precautions & Early Interventions</h4>
            <div class="advisory-item">
                <span style="font-size: 16px;">🩺</span>
                <div><strong>General Health Impact:</strong> {impact}</div>
            </div>
            <div class="advisory-item">
                <span style="font-size: 16px;">🏫</span>
                <div><strong>Schools & Children:</strong> {'Restrict prolonged outdoor games and heavy physical education.' if is_unhealthy else 'All outdoor sports and morning school activities can proceed safely.'}</div>
            </div>
            <div class="advisory-item">
                <span style="font-size: 16px;">👵</span>
                <div><strong>Vulnerable Groups (Asthma, Elderly, Cardiac):</strong> {'Keep rescue inhalers accessible; prefer exercising indoors.' if is_unhealthy else 'Safe for regular morning and evening strolls.'}</div>
            </div>
            <div class="advisory-item">
                <span style="font-size: 16px;">😷</span>
                <div><strong>Commuters & Masks:</strong> {'N95/FFP2 masks advised during morning peak traffic and industrial zones.' if pred_aqi > 200 else ('Light mask recommended for sensitive individuals.' if is_unhealthy else 'No mask required for general population.')}</div>
            </div>
            <div class="advisory-item">
                <span style="font-size: 16px;">🏠</span>
                <div><strong>Ventilation & Indoors:</strong> {'Keep windows closed during early morning temperature inversion; run HEPA air purifiers.' if pred_aqi > 150 else 'Ventilate rooms during midday fresh breeze.'}</div>
            </div>
        </div>
        """

        # ML Model vs Baseline Rule insight
        agreement = (is_unhealthy == baseline_flag)
        baseline_text = "Unhealthy (>100)" if baseline_flag else "Healthy (≤100)"
        ml_text = "Unhealthy (>100)" if is_unhealthy else "Healthy (≤100)"

        insight_html = f"""
        <div style="background:#f1f5f9; border-radius:12px; padding:14px 18px; font-size:13px; color:#334155; border:1px solid #cbd5e1;">
            <div style="font-weight:700; color:#0f172a; margin-bottom:6px; display:flex; align-items:center; gap:6px;">
                🤖 ML Model vs Baseline Rule Analysis
            </div>
            <div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:8px;">
                <div><strong>Baseline Rule (Today > 100):</strong> <span style="color:{'#ef4444' if baseline_flag else '#10b981'}; font-weight:600;">{baseline_text}</span></div>
                <div><strong>Random Forest ML Model:</strong> <span style="color:{'#ef4444' if is_unhealthy else '#10b981'}; font-weight:600;">{ml_text}</span></div>
                <div><strong>Status:</strong> {'✅ Models Agree' if agreement else '⚡ ML detected lag divergence from baseline rule'}</div>
            </div>
            <div style="font-size:12px; color:#64748b;">
                Unlike simple heuristics that assume tomorrow equals today, the Random Forest model incorporates multi-pollutant inertia (PM2.5: {pm25} µg/m³, NO2: {no2} µg/m³), temporal autocorrelation, and seasonal calendar patterns.
            </div>
        </div>
        """

        return banner_html, metrics_html, advisory_html, insight_html

    except Exception as e:
        err_msg = f"<div style='color:red; padding:15px; background:#fee2e2; border-radius:8px;'>Error generating forecast: {str(e)}</div>"
        return err_msg, "", "", ""


# Build Gradio UI Blocks
with gr.Blocks(title="Air Quality Forecasting & 24h Early Warning") as demo:
    # Header Banner
    gr.HTML("""
    <div class="main-header">
        <h1>Air Quality Forecasting & 24h Early Warning System</h1>
        <p>Predicting next-day unhealthy air (AQI &gt; 100) and continuous next-day AQI using multi-pollutant machine learning models.</p>
        <div class="header-badges">
            <span class="badge">🌲 Champion: Random Forest (F1: 0.9140)</span>
            <span class="badge">📉 RMSE: 32.96 vs 41.96 Baseline</span>
            <span class="badge">📍 74,402 Station Validation Records</span>
            <span class="badge">⚠️ 24-Hour Advance Alert</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # TAB 1: PREDICTOR
        with gr.Tab("🔮 24-Hour Early Warning Predictor"):
            with gr.Row():
                # LEFT: INPUT CONTROLS
                with gr.Column(scale=5):
                    gr.Markdown("### ⚙️ Input Pollutant Readings")

                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        label="⚡ Quick Load Scenario Presets",
                        value=None,
                        info="Click a preset to quickly populate realistic city air profiles"
                    )

                    with gr.Group():
                        gr.Markdown("**Core Readings (Primary Drivers)**")
                        with gr.Row():
                            aqi_input = gr.Slider(minimum=5, maximum=500, value=125, step=1, label="Today's Current AQI", info="Central Pollution Control Board index")
                            prev_aqi_input = gr.Slider(minimum=5, maximum=500, value=118, step=1, label="Yesterday's AQI (Temporal Lag)", info="Previous day reading")

                        with gr.Row():
                            pm25_input = gr.Slider(minimum=1.0, maximum=450.0, value=58.0, step=0.5, label="PM2.5 (µg/m³)", info="Fine particulate matter")
                            no2_input = gr.Slider(minimum=1.0, maximum=250.0, value=35.0, step=0.5, label="NO2 (µg/m³)", info="Nitrogen dioxide")
                            co_input = gr.Slider(minimum=0.05, maximum=15.0, value=1.1, step=0.05, label="CO (mg/m³)", info="Carbon monoxide")

                    with gr.Accordion("🔬 Advanced Pollutants & Temporal Features (Optional)", open=False):
                        gr.Markdown("<small style='color:#64748b'>Unspecified values automatically use median imputations from training data without data leakage.</small>")
                        with gr.Row():
                            pm10_input = gr.Slider(minimum=5.0, maximum=500.0, value=104.2, step=1.0, label="PM10 (µg/m³)")
                            so2_input = gr.Slider(minimum=0.5, maximum=120.0, value=11.2, step=0.5, label="SO2 (µg/m³)")
                            o3_input = gr.Slider(minimum=1.0, maximum=180.0, value=34.6, step=0.5, label="O3 (µg/m³)")

                        with gr.Row():
                            no_input = gr.Slider(minimum=0.5, maximum=150.0, value=13.5, step=0.5, label="NO (µg/m³)")
                            nox_input = gr.Slider(minimum=1.0, maximum=200.0, value=27.8, step=0.5, label="NOx (ppb)")

                        with gr.Row():
                            month_input = gr.Dropdown(choices=MONTH_NAMES, value="November", label="Month (Seasonality)", info="Winter months exhibit higher inversion")
                            day_input = gr.Dropdown(choices=DAY_NAMES, value="Wednesday", label="Day of Week", info="Accounts for weekend traffic patterns")

                    with gr.Row():
                        predict_btn = gr.Button("🚀 Forecast Tomorrow's Air Quality", variant="primary", scale=2)
                        reset_btn = gr.Button("🔄 Reset Defaults", scale=1)

                # RIGHT: RESULTS & ADVISORY
                with gr.Column(scale=6):
                    gr.Markdown("### 📊 Forecast Results & Early Warning")

                    out_banner = gr.HTML()
                    out_metrics = gr.HTML()
                    out_advisory = gr.HTML()
                    out_insights = gr.HTML()

            # Wire up interactions
            preset_dropdown.change(
                fn=apply_preset,
                inputs=[preset_dropdown],
                outputs=[
                    aqi_input, prev_aqi_input, pm25_input, no2_input, co_input,
                    pm10_input, no_input, nox_input, so2_input, o3_input,
                    month_input, day_input
                ]
            )

            all_inputs = [
                aqi_input, prev_aqi_input, pm25_input, no2_input, co_input,
                pm10_input, no_input, nox_input, so2_input, o3_input,
                month_input, day_input
            ]

            predict_btn.click(
                fn=run_forecast,
                inputs=all_inputs,
                outputs=[out_banner, out_metrics, out_advisory, out_insights]
            )

            reset_btn.click(
                fn=lambda: (100, 100, 50.0, 25.0, 1.0, 104.2, 13.5, 27.8, 11.2, 34.6, "June", "Thursday"),
                inputs=[],
                outputs=all_inputs
            )

            # Auto-run forecast on load
            demo.load(
                fn=run_forecast,
                inputs=all_inputs,
                outputs=[out_banner, out_metrics, out_advisory, out_insights]
            )

        # TAB 2: MODEL PERFORMANCE
        with gr.Tab("📈 Model Benchmarks & Comparison"):
            gr.Markdown("### 🏆 7-Model Evaluation Matrix (Held-out Test Set: 4,408 Days)")
            
            benchmark_data = {
                "Model Architecture": [
                    "Baseline Rule (No ML: Today's AQI > 100)",
                    "Logistic Regression",
                    "k-Nearest Neighbors (k-NN)",
                    "Decision Tree Classifier",
                    "Random Forest (Final Champion)",
                    "Support Vector Machine (SVM)",
                    "Gradient Boosting",
                    "Artificial Neural Network (MLP)"
                ],
                "Accuracy": [0.8761, 0.8814, 0.8736, 0.8505, 0.8963, 0.8879, 0.8934, 0.8907],
                "F1 Score": [0.8969, 0.9004, 0.8941, 0.8763, 0.9140, 0.9064, 0.9113, 0.9089],
                "ROC-AUC": [0.9398, 0.9504, 0.9338, 0.8434, 0.9619, 0.9509, 0.9599, 0.9597]
            }
            gr.DataFrame(value=pd.DataFrame(benchmark_data), interactive=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    #### 📉 Next-Day AQI Continuous Regression
                    - **Baseline (Tomorrow = Today):** RMSE `41.96`
                    - **Random Forest Regressor:** RMSE **`32.96`**
                    - *Improvement:* **21.5% reduction** in continuous prediction error over naive persistence.
                    """)
                with gr.Column():
                    gr.Markdown("""
                    #### 🛰️ External Validation on 74,402 Station Readings
                    - **Baseline Rule:** F1 `0.9184` | RMSE `47.22`
                    - **Saved ML Pipeline:** F1 **`0.9385`** | RMSE **`36.29`**
                    - Tested on completely unseen station-level monitoring points across India without retraining.
                    """)

            gr.Markdown("---")
            gr.Markdown("### 🧩 Pipeline Ablation Progression")
            ablation_data = {
                "Pipeline Configuration": [
                    "M1: Raw Pollutants Only (AQI, PM2.5, NO2, CO)",
                    "M2: M1 + Temporal Lag (Yesterday's AQI)",
                    "M3: M2 + Extra Pollutants + Calendar Seasonality (FINAL)"
                ],
                "Accuracy": [0.8841, 0.8863, 0.8963],
                "F1 Score": [0.9015, 0.9036, 0.9140],
                "Key Finding": [
                    "Baseline raw chemical readings",
                    "Temporal autoregression stabilizes next-day trajectory",
                    "Seasonality (Winter inversion) and extra pollutants yield biggest boost"
                ]
            }
            gr.DataFrame(value=pd.DataFrame(ablation_data), interactive=False)

        # TAB 3: GUIDELINES
        with gr.Tab("ℹ️ CPCB Guidelines & Methodology"):
            gr.Markdown("### 🇮🇳 Central Pollution Control Board (CPCB) AQI Standard")
            cpcb_table = {
                "AQI Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
                "Index Range": ["0 - 50", "51 - 100", "101 - 200", "201 - 300", "301 - 400", "401 - 500"],
                "Early Warning": ["Healthy", "Healthy", "⚠️ Unhealthy Trigger", "🚨 High Risk", "🛑 Severe Alert", "☣️ Emergency"],
                "Health Impact Summary": [
                    "Minimal impact; clean air.",
                    "Minor breathing discomfort to sensitive people.",
                    "Breathing discomfort to people with lung, asthma and heart disease.",
                    "Breathing discomfort to most people on prolonged exposure.",
                    "Respiratory illness on prolonged exposure; avoid outdoor sports.",
                    "Affects healthy people and seriously impacts those with existing diseases."
                ]
            }
            gr.DataFrame(value=pd.DataFrame(cpcb_table), interactive=False)

            gr.Markdown("""
            ---
            ### 🛠️ Methodological Highlights & Data Integrity
            1. **No Data Leakage:** Missing feature values are imputed solely using medians computed on the training set (`train_medians.pkl`). Scalers are strictly fitted on training splits only.
            2. **Temporal Splitting Rules:** Autoregressive shifts (`Previous AQI`, `Next AQI`) enforce strict consecutive day verification (`Date - Previous Date == 1 day`).
            3. **Outlier Boundaries:** Extreme sensor artifacts with AQI > 500 (beyond CPCB mathematical scale ceiling) are pruned.
            4. **Dual ML Objectives:**
               - **Classification:** Binary detection of next-day unhealthy air (`Next AQI > 100`) for early public health advisories.
               - **Regression:** Precise numeric AQI value forecasting for municipal trend tracking.
            """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft(), css=CUSTOM_CSS)

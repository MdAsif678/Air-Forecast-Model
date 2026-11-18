# ==============================================================================
# FASTAPI BACKEND SERVER: High-Density Utilitarian Air Quality Terminal
# ==============================================================================

import os
import uvicorn
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List

from predictor_engine import AQIPredictor, get_aqi_category_info, DEFAULT_MEDIANS

app = FastAPI(title="AeroCast Terminal - Utilitarian Air Quality System")

# Initialize the predictor engine
predictor = AQIPredictor()

os.makedirs("static", exist_ok=True)

class ForecastRequest(BaseModel):
    aqi: float
    prev_aqi: Optional[float] = None
    pm25: float
    no2: float
    co: float
    pm10: Optional[float] = None
    no: Optional[float] = None
    nox: Optional[float] = None
    so2: Optional[float] = None
    o3: Optional[float] = None
    month: Optional[int] = 11
    day_of_week: Optional[int] = 3


# EPA AQI Standard Flat Colors (No gradients)
EPA_COLORS = {
    "Good": {"bg": "#00E400", "text": "#000000", "border": "#00B000", "range": "0 - 50"},
    "Satisfactory": {"bg": "#92D050", "text": "#000000", "border": "#70A030", "range": "51 - 100"},
    "Moderate": {"bg": "#FFFF00", "text": "#000000", "border": "#D4D400", "range": "101 - 200"},
    "Poor": {"bg": "#FF7E00", "text": "#000000", "border": "#D96800", "range": "201 - 300"},
    "Very Poor": {"bg": "#FF0000", "text": "#FFFFFF", "border": "#CC0000", "range": "301 - 400"},
    "Severe": {"bg": "#7E0023", "text": "#FFFFFF", "border": "#550015", "range": "401 - 500"}
}

# 1-Click Terminal Scenarios
TERMINAL_PRESETS = [
    {
        "id": "clean_baseline",
        "code": "SCN-01:CLEAN",
        "label": "COASTAL / PRISTINE BASELINE",
        "location": "Kochi Station #04",
        "data": {
            "aqi": 38.0, "prev_aqi": 42.0, "pm25": 14.0, "no2": 8.5, "co": 0.40,
            "pm10": 32.0, "no": 4.2, "nox": 11.0, "so2": 5.1, "o3": 22.0,
            "month": 7, "day_of_week": 2
        }
    },
    {
        "id": "urban_satisfactory",
        "code": "SCN-02:SATIS",
        "label": "METRO SPRING BACKGROUND",
        "location": "Bengaluru BTM #01",
        "data": {
            "aqi": 76.0, "prev_aqi": 82.0, "pm25": 32.0, "no2": 21.0, "co": 0.75,
            "pm10": 68.0, "no": 10.5, "nox": 24.0, "so2": 8.4, "o3": 31.0,
            "month": 3, "day_of_week": 4
        }
    },
    {
        "id": "traffic_moderate",
        "code": "SCN-03:MOD-WARN",
        "label": "CONGESTED TRAFFIC CORRIDOR",
        "location": "Mumbai Sion #02",
        "data": {
            "aqi": 138.0, "prev_aqi": 125.0, "pm25": 65.0, "no2": 42.0, "co": 1.30,
            "pm10": 115.0, "no": 18.0, "nox": 38.0, "so2": 14.5, "o3": 44.0,
            "month": 10, "day_of_week": 1
        }
    },
    {
        "id": "industrial_poor",
        "code": "SCN-04:POOR",
        "label": "PRE-WINTER INDUSTRIAL SMOG",
        "location": "Kanpur Sharda #01",
        "data": {
            "aqi": 245.0, "prev_aqi": 210.0, "pm25": 118.0, "no2": 62.0, "co": 2.10,
            "pm10": 210.0, "no": 32.0, "nox": 65.0, "so2": 22.0, "o3": 58.0,
            "month": 11, "day_of_week": 3
        }
    },
    {
        "id": "inversion_severe",
        "code": "SCN-05:CRIT-HAZ",
        "label": "DELHI NCR WINTER INVERSION",
        "location": "Delhi Anand Vihar #03",
        "data": {
            "aqi": 412.0, "prev_aqi": 385.0, "pm25": 285.0, "no2": 98.0, "co": 3.90,
            "pm10": 390.0, "no": 55.0, "nox": 115.0, "so2": 34.0, "o3": 72.0,
            "month": 12, "day_of_week": 5
        }
    }
]

BENCHMARKS_TERMINAL = {
    "models": [
        {"name": "Baseline Persistence (Today > 100)", "family": "Heuristic Rule", "acc": "87.61%", "f1": "0.8969", "auc": "0.9398", "rank": "BASELINE"},
        {"name": "Logistic Regression (Class-Weighted)", "family": "GLM Linear", "acc": "88.14%", "f1": "0.9004", "auc": "0.9504", "rank": "EVALUATED"},
        {"name": "k-Nearest Neighbors (k=7)", "family": "Instance-Based", "acc": "87.36%", "f1": "0.8941", "auc": "0.9338", "rank": "EVALUATED"},
        {"name": "Decision Tree (CART)", "family": "Tree-Based", "acc": "85.05%", "f1": "0.8763", "auc": "0.8434", "rank": "EVALUATED"},
        {"name": "Random Forest (Selected Champion)", "family": "Ensemble Bagging", "acc": "89.63%", "f1": "0.9140", "auc": "0.9619", "rank": "CHAMPION"},
        {"name": "Support Vector Machine (RBF Kernel)", "family": "Kernel Methods", "acc": "88.79%", "f1": "0.9064", "auc": "0.9509", "rank": "EVALUATED"},
        {"name": "Gradient Boosting (GBDT)", "family": "Ensemble Boosting", "acc": "89.34%", "f1": "0.9113", "auc": "0.9599", "rank": "EVALUATED"},
        {"name": "Artificial Neural Network (MLP 2-Layer)", "family": "Feedforward NN", "acc": "89.07%", "f1": "0.9089", "auc": "0.9597", "rank": "EVALUATED"}
    ],
    "regression_metrics": {
        "baseline_rmse": "41.96 AQI pts",
        "champion_rmse": "32.96 AQI pts",
        "relative_error_reduction": "-21.45%"
    },
    "external_station_validation": {
        "test_records_count": 74402,
        "baseline_f1": "0.9184",
        "champion_f1": "0.9385",
        "baseline_rmse": "47.22",
        "champion_rmse": "36.29"
    },
    "ablation_stages": [
        {"stage": "STAGE-1: RAW POLLUTANTS", "features": "AQI, PM2.5, NO2, CO", "accuracy": "88.41%", "f1": "0.9015"},
        {"stage": "STAGE-2: TEMPORAL LAG", "features": "+ Yesterday's AQI (Autoregressive)", "accuracy": "88.63%", "f1": "0.9036"},
        {"stage": "STAGE-3: FULL SUITE + CALENDAR", "features": "+ PM10, SO2, O3, NO, NOx, Month, DayOfWeek", "accuracy": "89.63%", "f1": "0.9140"}
    ]
}


def compute_primary_driver(req: ForecastRequest):
    """Determine the primary pollutant exceeding regulatory ceilings."""
    ratios = [
        ("PM2.5", req.pm25, 60.0, "µg/m³"),
        ("PM10", req.pm10 or DEFAULT_MEDIANS["PM10"], 100.0, "µg/m³"),
        ("NO2", req.no2, 80.0, "µg/m³"),
        ("CO", req.co, 2.0, "mg/m³"),
        ("SO2", req.so2 or DEFAULT_MEDIANS["SO2"], 80.0, "µg/m³"),
        ("O3", req.o3 or DEFAULT_MEDIANS["O3"], 100.0, "µg/m³")
    ]
    # Rank by multiple of NAAQS 24h standard
    scored = []
    for name, val, limit, unit in ratios:
        multiple = val / limit
        scored.append((multiple, name, val, limit, unit))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[0]
    return {
        "pollutant": top[1],
        "value": round(top[2], 2),
        "limit": top[3],
        "unit": top[4],
        "pct_of_limit": round(top[0] * 100, 1),
        "exceeded": top[0] > 1.0,
        "all_ranked": [
            {"pollutant": s[1], "value": round(s[2], 2), "limit": s[3], "unit": s[4], "pct": round(s[0]*100, 1), "exceeded": s[0] > 1.0}
            for s in scored
        ]
    }


def generate_hourly_timeline(pred_aqi: float, current_aqi: float, pm25: float, no2: float, co: float):
    """Generate precise 24-hour step trajectory simulation from current to predicted AQI."""
    base_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timeline = []
    
    # 25 points from T+0 to T+24
    for h in range(25):
        t = base_time + timedelta(hours=h)
        weight = h / 24.0
        
        # Diurnal fluctuation curve (morning rush 8-10, evening rush 19-21, night inversion)
        hour_of_day = t.hour
        diurnal_factor = 1.0
        if 7 <= hour_of_day <= 10:
            diurnal_factor = 1.12
        elif 18 <= hour_of_day <= 22:
            diurnal_factor = 1.18
        elif 1 <= hour_of_day <= 5:
            diurnal_factor = 1.08  # night thermal inversion trapping
        elif 12 <= hour_of_day <= 16:
            diurnal_factor = 0.90  # daytime solar dispersion

        # Interpolate AQI
        base_interpolated = current_aqi + (pred_aqi - current_aqi) * weight
        h_aqi = round(max(5.0, min(500.0, base_interpolated * diurnal_factor)), 1)
        h_pm25 = round(max(2.0, (pm25 * (1 - weight) + (pm25 * (pred_aqi / max(current_aqi, 1.0))) * weight) * diurnal_factor), 1)
        h_no2 = round(max(2.0, (no2 * (1 - weight) + (no2 * (pred_aqi / max(current_aqi, 1.0))) * weight) * (1.15 if 8 <= hour_of_day <= 10 or 18 <= hour_of_day <= 21 else 0.95)), 1)
        h_co = round(max(0.1, (co * (1 - weight) + (co * (pred_aqi / max(current_aqi, 1.0))) * weight) * diurnal_factor), 2)
        h_unhealthy = (h_aqi > 100.0)

        timeline.append({
            "step_index": h,
            "hour_offset": f"+{h:02d}h",
            "iso_timestamp": t.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "display_time": t.strftime("%H:00 UTC"),
            "aqi": h_aqi,
            "pm25": h_pm25,
            "no2": h_no2,
            "co": h_co,
            "is_unhealthy": h_unhealthy,
            "flag": "UNHEALTHY" if h_unhealthy else "NORMAL"
        })
    return timeline


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


@app.get("/api/presets")
def get_presets():
    return JSONResponse(TERMINAL_PRESETS)


@app.get("/api/benchmarks")
def get_benchmarks():
    return JSONResponse(BENCHMARKS_TERMINAL)


@app.post("/api/predict")
def predict_telemetry(req: ForecastRequest):
    res = predictor.predict(
        aqi=req.aqi,
        prev_aqi=req.prev_aqi if req.prev_aqi is not None else req.aqi,
        pm25=req.pm25,
        no2=req.no2,
        co=req.co,
        pm10=req.pm10,
        no=req.no,
        nox=req.nox,
        so2=req.so2,
        o3=req.o3,
        month=req.month,
        day_of_week=req.day_of_week
    )

    pred_aqi = res["predicted_aqi"]
    is_unhealthy = res["is_unhealthy"]
    prob = res["unhealthy_prob_pct"]
    baseline_flag = res["baseline_rule_flag"]
    cat_name = res["category"]
    
    # Map to solid flat EPA colors
    epa_meta = EPA_COLORS.get(cat_name, EPA_COLORS["Moderate"])

    # Primary Pollutant calculation
    driver_info = compute_primary_driver(req)

    # 24-Hour Hourly Timeline Simulation
    timeline = generate_hourly_timeline(pred_aqi, req.aqi, req.pm25, req.no2, req.co)

    # Text warning banner
    if is_unhealthy:
        warning_banner = f"CRITICAL 24H EARLY WARNING: FORECAST AQI ({round(pred_aqi,1)}) BREACHES UNHEALTHY THRESHOLD (>100.0). PRIMARY DRIVER: {driver_info['pollutant']} ({driver_info['value']} {driver_info['unit']}, {driver_info['pct_of_limit']}% OF NAAQS CEILING)."
    else:
        warning_banner = f"NOMINAL: 24H FORECAST AQI ({round(pred_aqi,1)}) REMAINS WITHIN STATUTORY HEALTHY BOUNDS (<=100.0). PRIMARY POLLUTANT: {driver_info['pollutant']} ({driver_info['value']} {driver_info['unit']})."

    # Actionable Operational Protocols
    protocols = [
        {"code": "PROT-01:INSTITUTIONAL", "entity": "Schools & Child Care", "instruction": "SUSPEND all outdoor physical education and athletic training. Enforce indoor recess." if is_unhealthy else "Standard outdoor athletic curricula and morning assemblies authorized."},
        {"code": "PROT-02:CLINICAL", "entity": "Cardiopulmonary Cohorts", "instruction": "EMERGENCY PRECAUTION: Asthmatic and cardiac patients must remain in HEPA-filtered rooms. Keep rescue albuterol accessible." if is_unhealthy else "Routine outpatient activity permitted without clinical contraindication."},
        {"code": "PROT-03:OCCUPATIONAL", "entity": "Transit & Field Personnel", "instruction": "MANDATORY: Certified N95/FFP2 respirators required during transit. Limit unshielded road exposure." if pred_aqi > 200 else ("Particulate mask recommended during peak commute." if is_unhealthy else "Standard occupational exposure limits satisfied.")},
        {"code": "PROT-04:HVAC", "entity": "Facility Air Management", "instruction": "ISOLATE outdoor air intakes during morning thermal inversion (04:00-09:00 UTC). Run commercial HEPA filtration." if pred_aqi > 150 else "Maintain standard ASHRAE ventilation rates with ambient air exchange."}
    ]

    return JSONResponse({
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "predicted_aqi": pred_aqi,
        "current_aqi": req.aqi,
        "delta": res["aqi_delta"],
        "is_unhealthy": is_unhealthy,
        "confidence_pct": prob,
        "category": cat_name,
        "epa_colors": epa_meta,
        "primary_driver": driver_info,
        "warning_banner": warning_banner,
        "operational_protocols": protocols,
        "model_agreement": (is_unhealthy == baseline_flag),
        "baseline_says": "UNHEALTHY" if baseline_flag else "HEALTHY",
        "champion_says": "UNHEALTHY" if is_unhealthy else "HEALTHY",
        "timeline": timeline,
        "inputs": res["inputs"]
    })


@app.post("/api/export-csv")
def export_csv_data(req: ForecastRequest):
    res = predictor.predict(
        aqi=req.aqi, prev_aqi=req.prev_aqi if req.prev_aqi is not None else req.aqi,
        pm25=req.pm25, no2=req.no2, co=req.co, pm10=req.pm10, no=req.no,
        nox=req.nox, so2=req.so2, o3=req.o3, month=req.month, day_of_week=req.day_of_week
    )
    timeline = generate_hourly_timeline(res["predicted_aqi"], req.aqi, req.pm25, req.no2, req.co)
    
    # Generate CSV lines
    csv_lines = ["INDEX,HOUR_OFFSET,TIMESTAMP_UTC,AQI,PM2.5_UGM3,NO2_UGM3,CO_MGM3,ALERT_STATUS"]
    for row in timeline:
        csv_lines.append(f"{row['step_index']},{row['hour_offset']},{row['iso_timestamp']},{row['aqi']},{row['pm25']},{row['no2']},{row['co']},{row['flag']}")
    
    csv_content = "\n".join(csv_lines)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=aerocast_24h_telemetry.csv"}
    )


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=7860, reload=False)

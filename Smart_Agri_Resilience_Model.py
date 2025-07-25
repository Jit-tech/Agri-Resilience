# AI–Enabled Smart Agri Resilience Simulator (with Responses API)
# Author: Jit

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import statsmodels.api as sm
from openai import OpenAI

import os
# Initialize OpenAI Responses client, support Streamlit secrets or env var
openai_key = None
# Try reading from Streamlit secrets
try:
    openai_key = st.secrets["openai"]["api_key"]
except Exception:
    pass
# Fallback to environment variable
if not openai_key:
    openai_key = os.getenv("OPENAI_API_KEY", "")

if not openai_key:
    st.error("OpenAI API key not found. Set OPENAI_API_KEY in your environment or in Streamlit secrets.")

client = OpenAI(api_key=openai_key)  # Use the configured API key
model_name = "o4-mini"  # Set your OpenAI API key via Streamlit secrets
model_name = "o4-mini"

# Streamlit config
st.set_page_config(page_title="Smart Agri Simulator", layout="wide")

# Sidebar: CSV uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset", type=["csv"], key="file_uploader"
)

# Cache data loader
def load_csv(file):
    if file is not None:
        return pd.read_csv(file, parse_dates=True)
    return None
user_data = st.cache_data(load_csv)(uploaded_file)

# Initialize session state
def init_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
init_state()

# UI Tabs
tabs = st.tabs([
    "Crops", "Livestocks", "AI Agent", "Supply Chain", "SDG Overlay"
])

##### Plants Tab #####
with tabs[0]:
    st.header("Crop Monitoring & Forecasting")
    plant_type = st.selectbox(
        "Plant Type", ["Potato", "Wheat", "Barley", "Maize"], key="plant_type"
    )
    canopy_temp = st.slider(
        "Canopy Temperature (°C)", 10, 45, 25, key="canopy_temp"
    )
    soil_moisture = st.slider(
        "Soil Moisture (%)", 0, 100, 45, key="soil_moisture"
    )
    sap_flow = st.slider(
        "Sap Flow Rate (ml/hr)", 0, 500, 250, key="sap_flow"
    )
    chlorophyll_index = st.slider(
        "Chlorophyll Index (0-1)", 0.0, 1.0, 0.6, key="chloro_index"
    )
    crispr_boost = st.checkbox(
        "Apply CRISPR-enhanced Trait", key="crispr_boost"
    )
    climate_shock = st.selectbox(
        "Climate Scenario", ["None", "Heatwave", "Drought", "Flood"], key="climate_shock"
    )

    # Computations
    boost = 0.1 if crispr_boost else 0.0
    plant_stress = float(
        np.clip(
            1 - soil_moisture / 100 * 0.3 - sap_flow / 500 * 0.3 - chlorophyll_index * 0.4 - boost,
            0,
            1,
        )
    )
    loss_map = {"None": 0, "Heatwave": 15, "Drought": 25, "Flood": 20}
    predicted_yield = float(100 - (loss_map[climate_shock] + plant_stress * 30))

    st.metric("Crop Stress Index", f"{plant_stress:.2f}")
    st.metric("Predicted Yield (%)", f"{predicted_yield:.2f}")
    st.caption("Stress index = moisture + flow + chlorophyll + CRISPR boost.")

    # Econometric model
    if user_data is not None and 'Yield' in user_data.columns:
        st.subheader("Actual vs Predicted Yield")
        df = user_data.copy()
        X = df.drop(columns=['Yield', 'Date'], errors='ignore')
        X = sm.add_constant(X)
        y = df['Yield']
        model = sm.OLS(y, X).fit()
        df['ForecastYield'] = model.predict(X)
        fig = px.line(
            df,
            x='Date' if 'Date' in df.columns else df.index,
            y=['Yield', 'ForecastYield'],
            labels={'value': 'Yield (%)', 'Date': 'Date'},
            title='Crop Yield: Actual vs Predicted',
        )
        fig.update_traces(mode='lines+markers')
        fig.update_traces(
            selector={'name': 'ForecastYield'},
            line={'dash': 'dash', 'color': 'orange'},
        )
        fig.update_layout(legend_title_text='Legend')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Interpretation:** Solid = Actual; Dashed orange = Predicted via econometric model."
        )

##### Animals Tab #####
with tabs[1]:
    st.header("Livestock Health & Yield Forecasting")
    animal_type = st.selectbox(
        "Animal Type", ["Dairy Cow", "Sheep", "Goat", "Beef Cattle"], key="animal_type"
    )
    body_temp = st.slider(
        "Body Temperature (°C)", 35.0, 42.0, 38.5, key="body_temp"
    )
    gait = st.slider(
        "Gait Stability (0-1)", 0.0, 1.0, 0.8, key="gait"
    )
    feed_intake = st.slider(
        "Feed Intake (kg/day)", 0, 50, 25, key="feed_intake"
    )
    heart_var = st.slider(
        "Heart Rate Var (ms)", 20, 120, 65, key="heart_var"
    )
    elite_breed = st.checkbox(
        "Elite Breed Genetics", key="elite_breed"
    )

    boost_a = 0.1 if elite_breed else 0.0
    welfare_index = float(
        np.clip(
            0.25 * (42 - body_temp)
            + 0.25 * gait
            + 0.25 * (feed_intake / 50)
            + 0.25 * (heart_var / 120)
            + boost_a,
            0,
            1,
        )
    )
    methane = float(500 - (feed_intake * 5 + gait * 100))
    milk_yield = float(feed_intake * (welfare_index + 0.3))

    st.metric("Livestock Welfare Index", f"{welfare_index:.2f}")
    st.metric("Methane Emission (g/day)", f"{methane:.2f}")
    st.metric("Predicted Milk Yield (L/day)", f"{milk_yield:.2f}")
    st.caption("Welfare index integrates physiological & activity metrics.")

    if user_data is not None and 'MilkYield' in user_data.columns:
        st.subheader("Actual vs Predicted Milk Yield")
        df = user_data.copy()
        X = df.drop(columns=['MilkYield', 'Date'], errors='ignore')
        X = sm.add_constant(X)
        y = df['MilkYield']
        model = sm.OLS(y, X).fit()
        df['ForecastMilk'] = model.predict(X)
        fig = px.line(
            df,
            x='Date' if 'Date' in df.columns else df.index,
            y=['MilkYield', 'ForecastMilk'],
            labels={'value': 'Milk Yield (L)', 'Date': 'Date'},
            title='Milk Yield: Actual vs Predicted',
        )
        fig.update_traces(mode='lines+markers')
        fig.update_traces(
            selector={'name': 'ForecastMilk'},
            line={'dash': 'dash', 'color': 'orange'},
        )
        fig.update_layout(legend_title_text='Legend')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Interpretation:** Solid = Actual; Dashed orange = Predicted via econometric model."
        )

##### Agents Tab #####
with tabs[2]:
    st.header("Agentic AI Conversational Interface")
    # Use Responses API for dynamic chat
    user_query = st.text_input("Ask a question about crop or livestock scenarios:", key="agent_input")
    if user_query:
        try:
            response = client.responses.create(
                model=model_name,
                input=user_query,
                store=False
            )
            st.subheader("AI Response")
            st.write(response.output_text)
        except Exception as e:
            st.error(f"AI query failed: {e}")

##### Supply Chain Tab #####
with tabs[3]:
    st.header("Supply Chain & Market Impact")
    delay = st.slider("Transport Delay (days)", 0, 14, 2, key="delay")
    cold_chain = st.checkbox("Enable Cold Chain", key="cold_chain")
    disruption = st.selectbox(
        "Disruption Type",
        ["None", "Strike", "Weather", "Geopolitical"],
        key="disruption",
    )
    penalty_map = {"None": 0, "Strike": 12, "Weather": 8, "Geopolitical": 15}
    penalty = penalty_map[disruption]
    cooling_adj = -5 if cold_chain else 0
    risk = delay * 3 + penalty + cooling_adj
    price_drop = delay * 2 + penalty * 0.5

    st.metric("Loss Risk (%)", f"{risk:.2f}")
    st.metric("Market Price Drop (%)", f"{price_drop:.2f}")
    st.caption("Adjust parameters to simulate supply chain impacts.")

##### SDG Overlay Tab #####
with tabs[4]:
    st.header("SDG & CAP Policy Alignment")
    sdg2 = float(100 - plant_stress * 100)
    sdg13 = float(100 - methane / 5)
    focus = st.radio(
        "Policy Focus",
        ["SDG2", "SDG13", "Balanced"],
        key="policy_focus",
    )
    if focus == "SDG2":
        cap_score = sdg2
    elif focus == "SDG13":
        cap_score = sdg13
    else:
        cap_score = round((sdg2 + sdg13) / 2, 2)
    cap_score += (int(crispr_boost) + int(elite_breed)) * 5

    st.metric("SDG2 Alignment", f"{sdg2:.2f}%")
    st.metric("SDG13 Alignment", f"{sdg13:.2f}%")
    st.success(f"CAP Subsidy Score: {cap_score:.2f}/100")

# Sidebar instructions
st.sidebar.title("Smart Agri Resilience Simulator")
st.sidebar.markdown(
    "Developed by Jit \n"
    "   (Powered by Econometrics and Wearable Sensor Data) \n")
    "   (Agentic AI is powered by Open AI")
)

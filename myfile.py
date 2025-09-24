import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pickle
import matplotlib.pyplot as plt
import os
import joblib


# ----------------------------
# Load Saved Model
# ----------------------------
MODEL_PATH = "fish_catch_model.pkl"
model = joblib.load(MODEL_PATH)


# ----------------------------
# Species Options (from dataset)
# ----------------------------
species_list = [
    "Acanthocybium solandri",
    "Acetes",
    "Alopias",
    "Anguilliformes",
    "Anodontostoma chacunda",
    "Ariidae",
    "Auxis",
    "Auxis rochei",
    "Auxis thazard",
    "Batoidea",
    "Bivalvia",
    "Bohadschia marmorata",
    "Bramidae",
    "Bregmaceros mcclellandi",
    "Carangidae",
    "Caranx",
    "Carcharhinidae",
    "Carcharhinus falciformis",
    "Carcharhinus longimanus",
    "Carcharhinus obscurus",
    "Cephalopoda",
    "Cephea",
    "Chirocentrus",
    "Chirocentrus dorab",
    "Chondrichthyes",
    "Clupeidae",
    "Clupeiformes",
    "Congridae",
    "Coryphaena hippurus",
    "Crassostrea",
    "Decapoda",
    "Decapterus",
    "Decapterus russelli",
    "Dendrobranchiata",
    "Elagatis bipinnulata",
    "Elasmobranchii",
    "Engraulidae",
    "Epinephelus",
    "Euthynnus affinis",
    "Fenneropenaeus merguiensis",
    "Gastropoda",
    "Gymnosarda unicolor",
    "Harpadon nehereus",
    "Hemiramphus",
    "Hilsa kelee",
    "Holothuria atra",
    "Holothuria edulis",
    "Holothuroidea",
    "Istiompax indica",
    "Istiophoridae",
    "Istiophorus",
    "Istiophorus platypterus",
    "Isurus",
    "Isurus oxyrinchus",
    "Kajikia audax",
    "Katsuwonus pelamis",
    "Lactarius lactarius",
    "Lates calcarifer",
    "Leiognathidae",
    "Leiognathus",
    "Lepidocybium flavobrunneum",
    "Lethrinidae",
    "Loliginidae",
    "Lutjanidae",
    "Makaira",
    "Manta birostris",
    "Marine fishes not identified",
    "Marine pelagic fishes not identified",
    "Megalaspis cordyla",
    "Metapenaeus",
    "Miscellaneous aquatic invertebrates",
    "Miscellaneous marine crustaceans",
    "Modiolus",
    "Mollusca",
    "Monacanthidae",
    "Mugil",
    "Mugilidae",
    "Mullidae",
    "Muraenesox",
    "Muraenesox cinereus",
    "Mytilidae",
    "Nemipteridae",
    "Nemipterus",
    "Octopodidae",
    "Octopus",
    "Palinuridae",
    "Pampus",
    "Pampus argenteus",
    "Pampus chinensis",
    "Parastromateus niger",
    "Pellona ditchela",
    "Penaeidae",
    "Penaeus",
    "Penaeus monodon",
    "Penaeus semisulcatus",
    "Pennahia",
    "Perciformes",
    "Perna viridis",
    "Placuna placenta",
    "Platycephalidae",
    "Pleuronectiformes",
    "Plotosidae",
    "Polynemidae",
    "Polynemus",
    "Portunus pelagicus",
    "Priacanthidae",
    "Priacanthus",
    "Prionace glauca",
    "Psettodes erumei",
    "Pteriomorphia",
    "Rajiformes",
    "Rastrelliger",
    "Rastrelliger kanagurta",
    "Rhizostomeae",
    "Sarda orientalis",
    "Sardinella",
    "Sardinella longiceps",
    "Saurida",
    "Sciaenidae",
    "Scolopsis",
    "Scomberoides",
    "Scomberomorus",
    "Scomberomorus commerson",
    "Scomberomorus guttatus",
    "Scomberomorus lineolatus",
    "Scombridae",
    "Scombroidei",
    "Scylla serrata",
    "Selar crumenophthalmus",
    "Sepia",
    "Sepiidae",
    "Sepioteuthis lessoniana",
    "Sergestidae",
    "Seriolina nigrofasciata",
    "Serranidae",
    "Sillaginidae",
    "Sillago",
    "Siluriformes",
    "Soleidae",
    "Sparidae",
    "Sphyraena",
    "Sphyraena barracuda",
    "Sphyraenidae",
    "Sphyrna",
    "Sphyrna lewini",
    "Sphyrnidae",
    "Stolephorus",
    "Stromateidae",
    "Synodontidae",
    "Tegillarca granosa",
    "Tenualosa ilisha",
    "Tetrapturus angustirostris",
    "Teuthida",
    "Thenus orientalis",
    "Thryssa",
    "Thunnus",
    "Thunnus alalunga",
    "Thunnus albacares",
    "Thunnus obesus",
    "Thunnus tonggol",
    "Trichiuridae",
    "Trichiurus",
    "Trichiurus lepturus",
    "Veneridae",
    "Xiphias gladius",
    "Zeus faber"
]  # extend this with all unique values from your dataset


# Month options
month_list = list(range(1, 13))

# ----------------------------
# Sidebar for Input Parameters
# ----------------------------
st.sidebar.title("Fish Catch Prediction Parameters") 
selected_species = st.sidebar.selectbox("Select Species (scientific_name)", species_list) 
selected_year = st.sidebar.number_input("Year", min_value=2025, max_value=2100, value=2025, step=1) 
selected_month = st.sidebar.selectbox("Select Month", month_list)

# Environmental + lag inputs
chlorophyll = st.sidebar.number_input("Chlorophyll", min_value=0.0, value=2.0, step=0.1)
sst = st.sidebar.number_input("Sea Surface Temperature (°C)", min_value=0.0, value=25.0, step=0.1)
salinity = st.sidebar.number_input("Salinity", min_value=0.0, value=32.0, step=0.1)
ocean_currents = st.sidebar.number_input("Ocean Currents", min_value=0.0, value=1.5, step=0.1)
lag_1 = st.sidebar.number_input("Lag_1", min_value=0.0, value=100.0, step=1.0)
lag_2 = st.sidebar.number_input("Lag_2", min_value=0.0, value=80.0, step=1.0)
lag_3 = st.sidebar.number_input("Lag_3", min_value=0.0, value=60.0, step=1.0)

# ----------------------------
# Display Selections
# ----------------------------
st.markdown(
    "<h2 style='color:#1e40af; text-align:center; font-size:70px; font-weight:700;'>DeepFin</h2>",
    unsafe_allow_html=True
)
st.title("Fish Catch Prediction")
st.write(f"Species: {selected_species}")
st.write(f"Year: {selected_year}")
st.write(f"Month: {selected_month}")

# ----------------------------
# Single Prediction
# ----------------------------
input_df = pd.DataFrame([{
    "scientific_name": selected_species,
    "year": selected_year,
    "month": selected_month,
    "chlorophyll": chlorophyll,
    "sst": sst,
    "salinity": salinity,
    "ocean_currents": ocean_currents,
    "lag_1": lag_1,
    "lag_2": lag_2,
    "lag_3": lag_3
}])

try:
    prediction = model.predict(input_df)[0]
    st.success(f"### Predicted Catch: {prediction:.2f} tonnes")
except Exception as e:
    st.error(f"Prediction failed: {e}")
    prediction = None
    
# ----------------------------
# Forecast Next 6 Years
# ----------------------------
if prediction is not None:
    years = np.arange(selected_year, selected_year + 6)
    pred_values = np.maximum(0, prediction * (1 + 0.05 * np.arange(6)))

    forecast_df = pd.DataFrame({
        "Year": years,
        "Predicted Catch": pred_values
    })

    st.subheader("📊 Predicted Catch for Next 6 Years")
    st.bar_chart(forecast_df.set_index("Year"))

    # Conditional Warnings
    if prediction > 1000:
        st.error("⚠ Overfishing Risk! Predicted catch is very high.")
    elif prediction > 500:
        st.warning("⚠ Moderate fishing pressure detected.")
    else:
        st.success("✅ Sustainable catch level predicted.")


# ----------------------------
# Batch Predictions
# ----------------------------
st.subheader("📂 Upload CSV for Batch Predictions")
uploaded_file = st.file_uploader("Upload CSV with columns: scientific_name, year, month, chlorophyll, sst, salinity, ocean_currents, lag_1, lag_2, lag_3", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    try:
        batch_preds = model.predict(batch_df)
        batch_df["Predicted_Tonnes"] = batch_preds
        st.write("### Batch Prediction Results")
        st.dataframe(batch_df)
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

# ----------------------------
# Environmental Factors Visualization
# ----------------------------
st.subheader("🌍 Environmental Factors vs Catch Predictions")

env_months = pd.date_range(start=f"{selected_year}-{selected_month:02d}-01", periods=6, freq="M")
sst_vals = np.random.uniform(22, 30, size=6)  # Dummy SST
chl_vals = np.random.uniform(0.4, 2.5, size=6)  # Dummy Chlorophyll

fig, ax1 = plt.subplots()
ax1.plot(env_months, pred_values if prediction is not None else np.zeros(6), marker="o", label="Predicted Catch", color="blue")
ax1.set_ylabel("Catch (tonnes)", color="blue")

ax2 = ax1.twinx()
ax2.plot(env_months, sst_vals, marker="s", label="SST", color="red")
ax2.plot(env_months, chl_vals, marker="^", label="Chlorophyll", color="green")
ax2.set_ylabel("Environmental Factors", color="green")

fig.legend(loc="upper left")
st.pyplot(fig)

st.title("🌊 Demo Oceanographic Features Visualization")

# ----------------------------
# Generate Dummy Oceanographic Data
# ----------------------------
np.random.seed(42)

# Define months and ocean zones
months = pd.date_range(start="2025-01-01", periods=12, freq="M")
zones = ["Zone 1", "Zone 2", "Zone 3"]

# Create synthetic data for all months and zones
data_list = []
for zone in zones:
    for month in months:
        data_list.append({
            "Month": month,
            "Zone": zone,
            "SST (°C)": np.random.uniform(18, 30),
            "Chlorophyll (mg/m³)": np.random.uniform(0.5, 5.0),
            "Salinity (PSU)": np.random.uniform(30, 37),
            "Ocean Currents (m/s)": np.random.uniform(0, 2)
        })

ocean_data = pd.DataFrame(data_list)

# ----------------------------
# Interaction Widgets
# ----------------------------
selected_zone = st.selectbox("Select Ocean Zone", zones)
selected_parameter = st.selectbox(
    "Select Oceanographic Parameter to Plot",
    ["SST (°C)", "Chlorophyll (mg/m³)", "Salinity (PSU)", "Ocean Currents (m/s)"]
)

# Filter data based on selected zone
filtered_data = ocean_data[ocean_data["Zone"] == selected_zone]

# ----------------------------
# Plotting
# ----------------------------
chart = alt.Chart(filtered_data).mark_line(point=True).encode(
    x=alt.X('Month:T', title='Month'),
    y=alt.Y(f'{selected_parameter}:Q', title=selected_parameter),
    tooltip=['Month', selected_parameter]
).interactive()

st.altair_chart(chart, use_container_width=True)

# ------------------ Load model ------------------


# ------------------------- Load saved models/metadata -------------------------
PICKLE_PATH = "prophet_species_models (1).pkl"
PLOTS_DIR = "species_plots"
SIX_YR_PLOTS_DIR = "species_6yr_plots"   # NEW: for 6-year precomputed plots

@st.cache_resource
def load_saved():
    with open(PICKLE_PATH, 'rb') as f:
        saved = pickle.load(f)
    return saved

saved = load_saved()
species_list = saved.get('species_trained', [])
regressor_cols = saved.get('regressor_cols', [])

st.title("Overfishing Prediction")
st.sidebar.markdown("---")
st.sidebar.header("Overfishing Prediction Parameters")

# Sidebar controls
selected_species = st.sidebar.selectbox("Species", species_list)
selected_year = st.sidebar.number_input("Forecast year", min_value=2022, max_value=2035, value=2025)
selected_zone = st.sidebar.selectbox("Zone", options=["4 → Gujarat", "51 → West Coast", "57 → Kerala/Lakshadweep", "58/West Coast → Tamil Nadu/Bay"])   # update if needed


# ------------------------- Show saved metrics & prediction -------------------------
if selected_species not in saved['models_info']:
    st.warning("No trained model for selected species (insufficient historical years).")
    st.stop()

info = saved['models_info'][selected_species]

# Display selections on dashboard (vertical layout)
st.subheader("Selected Parameters")
st.write(f"**Species:** {selected_species}")
st.write(f"**Year:** {selected_year}")
st.write(f"**Zone:** {selected_zone}")

# Show predicted next-year & MSY
pred_next = info['predicted_next_year']
msy = info['msy']
ratio = info['exploitation_ratio']
flag = info['overfishing_flag']

# ------------------------- Interactive re-prediction with modified regressors -------------------------
st.markdown("---")
st.subheader("📌 Selected Year Prediction")

model = info['model']   # Prophet model

# Build regressor row
pred_row = {r: 0 for r in regressor_cols}

# Apply Zone and Status
zone_col = f"area_code_{selected_zone}"
if zone_col in pred_row:
    pred_row[zone_col] = 1

# Build df for prediction
ds_val = pd.to_datetime(f"{selected_year}-01-01")
pred_df = pd.DataFrame([{**pred_row, 'ds': ds_val}])[['ds'] + regressor_cols]

# Run prediction
try:
    forecast = model.predict(pred_df)
    yhat = float(forecast['yhat'].iloc[0])
    yhat_lower = float(forecast['yhat_lower'].iloc[0])
    yhat_upper = float(forecast['yhat_upper'].iloc[0])

    ratio2 = yhat / (msy if msy != 0 else 1)
    status_text = (
        "<span style='color:red; font-weight:bold;'>Overfished</span>" if ratio2 > 1.0 else
        "<span style='color:orange; font-weight:bold;'>Fully fished</span>" if ratio2 > 0.8 else
        "<span style='color:green; font-weight:bold;'>Sustainable</span>"
    )

    # --------- Show results in styled text ---------
    st.markdown(
        f"""
        <div style="font-size:18px; line-height:1.8;">
            <b>Predicted catch ({selected_year}):</b> {yhat:,.1f} tonnes <br>
            <b>95% Confidence Interval:</b> {yhat_lower:,.1f} - {yhat_upper:,.1f} tonnes <br>
            <b>MSY (proxy):</b> {msy:,.1f} tonnes <br>
            <b>Exploitation ratio:</b> {ratio2:.2f} <br>
            <b>Status:</b> {status_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------- Visualization ----------------
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([selected_year], [yhat], color="steelblue", label="Predicted Catch")
    ax.errorbar([selected_year], [yhat],
                yerr=[[yhat - yhat_lower], [yhat_upper - yhat]],
                fmt='o', color='black', capsize=5, label="95% CI")
    ax.axhline(msy, color="red", linestyle="--", label="MSY")
    ax.set_ylabel("Catch (tonnes)")
    ax.set_xlabel("Year")
    ax.set_title(f"Interactive Forecast: {selected_species} ({selected_year})")
    ax.legend()
    st.pyplot(fig)
    # -----------------------------------------------

except Exception as e:
    st.error(f"Interactive prediction failed: {e}")


# ------------------------- 6-Year Forecast Visualization -------------------------
st.markdown("---")
st.subheader("📊 Future 6-Year Prediction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIX_YR_PLOTS_DIR = os.path.join(BASE_DIR, "species_plots (2)", "species_plots")

six_yr_plot_path = f"{SIX_YR_PLOTS_DIR}/{selected_species}_forecast.png"
six_yr_plot_path = os.path.join(SIX_YR_PLOTS_DIR, f"{selected_species}_forecast.png")

 #Compute status (reuse ratio and msy already defined above)
st.markdown(
    f"""
    <div style="font-size:18px; line-height:1.8; margin-bottom:15px;">
        <b>Predicted next-year catch:</b> {pred_next:,.1f} tonnes <br>
        <b>MSY (proxy):</b> {msy:,.1f} tonnes <br>
        <b>Exploitation ratio:</b> {ratio:.2f} <br>
        <b>Overfishing flag:</b> {"<span style='color:red; font-weight:bold;'>YES</span>" if flag else "<span style='color:green; font-weight:bold;'>NO</span>"}
    </div>
    """,
    unsafe_allow_html=True
)

# Show image if available
if os.path.exists(six_yr_plot_path):
    st.image(six_yr_plot_path, caption=f"{selected_species} 6-Year Forecast", use_column_width=True)
else:
    st.info("No 6-year forecast plot available for this species.")

st.markdown("----")

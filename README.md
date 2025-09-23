# 🌊 AI-Driven Unified Data Platform for Fisheries & Ocean Insights  

## 🔹 Problem Statement

AI-Driven Unified Data Platform for Oceanographic, Fisheries, and Molecular Biodiversity Insights (SIH25041)

---

## 🔹 Project Overview  

We are building an **AI-driven unified platform** to predict fish catch by species and fishing zones, while linking it with oceanographic parameters such as **temperature, salinity, chlorophyll, and currents**.  
The goal is to create a **scalable and accessible system** that can support scientists, policymakers, and fishers in making sustainable decisions.  

---

## 🔹 Key Elements  

- **Integration**: Combines fisheries catch data (FAO) with oceanographic datasets (SAU) and keeps room for future biodiversity/eDNA modules.  
- **Prediction & Visualization**: Uses ML/statistical models (**XGBoost, Prophet, regression**) to forecast catches and highlight **“Safe-to-Fish Zones.”**  
- **Overfishing Alerts**: Flags catches above sustainable thresholds for decision support.  
- **Extensibility**: The model can be easily extended to a **frontend dashboard using Streamlit** for interactive visualization and user-friendly access.  
- **Scalable & Modular**: Designed to easily include new data sources in the future.  

---

## 🔹 Our Approach  

- Started with **species-wise catch prediction** using Sea Around Us landings.  
- Linked **oceanographic correlations** (SST, salinity, chlorophyll, currents) from FAO and synthetic datasets.  
- Implemented **risk alerts** for overfishing conditions.  
- Designed **visual outputs** combining predictions, ocean parameters, and alerts.  
- Maintains room for **future integration** of molecular/biodiversity (**eDNA**) modules and frontend dashboards.  

---

## 🔹 Datasets  

- **Sea Around Us – India EEZ Data** – Marine catch and effort data for India’s Exclusive Economic Zone.  
- **Food and Agriculture Organization (FAO)** – Annual global capture production data.  
- **Synthetic Data** – Created for missing features to simulate real-world scenarios.  

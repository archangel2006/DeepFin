# 🌊 AI-Driven Unified Data Platform for Fisheries & Ocean Insights  

## 🔹 Problem Statement

AI-Driven Unified Data Platform for Oceanographic, Fisheries, and Molecular Biodiversity Insights (SIH25041)

---

## 🔹 Project Overview  

We are building an **AI-driven unified platform** to predict fish catch by species and fishing zones, while linking it with oceanographic parameters such as **temperature, salinity, chlorophyll, and currents**.  
Our goal is to create a **scalable and accessible system** that can support scientists, policymakers, and fishers in making sustainable decisions.  

---

## 🔹 Key Elements  

- **Integration**: Bringing together fisheries catch data (FAO) with oceanographic datasets (SAU) and leaving scope for future biodiversity/eDNA modules.  
- **Prediction & Visualization**: Applying ML/statistical models (**XGBoost, Prophet, regression**) to forecast catches and display **“Safe-to-Fish Zones.”**  
- **Scalability**: Modular backend that allows quick ingestion of new data sources.  
- **Usability**: Intuitive dashboard with maps, time-series plots, alert banners, and data downloads.  
- **APIs & Docs**: REST APIs for predictions and data inputs with proper documentation.  

---

## 🔹 Our Approach  

- Began with **species-wise catch prediction** using Sea Around Us landings.  
- Incorporated **oceanographic correlations** (SST, salinity, chlorophyll, currents) from FAO and synthetic datasets where needed.  
- Added **overfishing risk alerts** to flag catches above sustainable thresholds.  
- Designed **visualizations** that combine predictions, ocean parameters, and risk indicators.  
- Left room for **future expansion** with molecular/biodiversity (**eDNA**) modules.  

---

## 🔹 Datasets  

- **Sea Around Us – India EEZ Data** – Marine catch and effort data for India’s Exclusive Economic Zone (EEZ).
- **Food and Agriculture Organization (FAO)** – Annual global capture production data.  
- **Synthetic Data** – generated where real features are unavailable.  

---

# UIDAI Data Hackathon 2026

## Project: India's Invisible Citizens - Bridging Aadhaar Exclusion Zones

**Author:** Divyanshu Patel  

---

## 1. Problem Statement
Despite Aadhaar's massive success (1.3B+ enrollments), significant "Exclusion Zones" persist where coverage lags, particularly for:
- **Children (0-5 years):** 3.5M+ estimated missing or needing updates.
- **Remote/Tribal Areas:** 174 districts identified as high-risk.
- **Migrant Populations:** High biometric failure rates (8.7% in tribal areas vs 2.1% urban).

These invisible citizens are often the most in need of welfare services but face the highest barriers to access.

## 2. Our Data-Driven Approach
We analyzed **5.4 million+ records** (Enrolment, Demographic, Biometric) using a 4-stage pipeline:

1.  **Exploratory Analysis:** Mapped exclusion hotspots (Odisha, Jharkhand, Northeast).
2.  **Predictive Modeling:** Trained a Gradient Boosting Classifier (99% accuracy) to predict exclusion risk based on migration intensity and child population.
3.  **Intervention Strategy:** Designed a cost-benefit optimized plan to deploy Mobile Enrollment Units (MEUs).
4.  **Impact Projection:** Calculated economic ROI and social benefits.

## 3. Key Findings & Impact
- **Identified 174 High-Risk Districts:** Prioritized top 100 for immediate intervention.
- **Solution:** Deploy **100 Mobile Enrollment Units (MEUs)** in a phased 21-month rollout.
- **Cost:** ₹7.25 Crores total investment.
- **Benefit:** **₹254.01 Crores** economic benefit (10-year NPV).
- **ROI:** **35:1 Benefit-Cost Ratio** (3,404% ROI).
- **Reach:** 450,000+ citizens brought into the digital identity fold.

---

## 4. Repository Structure

```
gov/
├── README.md                          # Project overview (this file)
├── UIDAI_official_instructions.txt    # Reference instructions
│
├── dataset/                           # Raw Data Sources (5M+ records)
│   ├── api_data_aadhar_enrolment/
│   ├── api_data_aadhar_demographic/
│   └── api_data_aadhar_biometric/
│
├── notebooks/                         # Analysis Notebooks (Source Code)
│   ├── 01_data_preparation.ipynb      # ETL Pipeline
│   ├── 02_exploratory_analysis.ipynb  # EDA & Visualization
│   ├── 03_exclusion_modeling.ipynb    # Machine Learning Model
│   ├── 04_intervention_strategy.ipynb # Optimization & Financials
│   └── 05_visualization_report.ipynb  # Dashboard Generation
│
├── outputs/                           # Generated Artifacts
│
└── src/                               # Helper Scripts
    
```

---

## 5. How to Reproduce
1.  **Environment:** Python 3.9+, Jupyter Lab.
2.  **Install Dependencies:** `pip install -r requirements.txt`
3.  **Run Notebooks:** Execute notebooks `01` through `05` in order. The outputs will populate in the `outputs/` folder.

## 6. Team Information
- **Lead:** Divyanshu Patel
- **Contact:** [itzdivyanshupatel@gmail.com]
- **Role:** Data Scientist & Policy Analyst

---
*Submitted for UIDAI Data Hackathon 2026*
# AI Cricket Analysis — `cricket_data` Module

This repository is part of the broader **AI Cricket Analysis** project and serves as the core data module. It contains all raw and processed cricket datasets under the `cricket_data` folder, along with the necessary scripts for data ingestion, cleaning, transformation, and feature engineering.


## 🎯 Project Overview

The goal of the **AI Cricket Analysis** project is to apply data science, machine learning, and statistical modeling to cricket match data to derive deep insights, make robust predictions, and visualize performance trends.

The **`cricket_data`** module specifically handles:
* **Data Ingestion:** Parsing and loading raw match data (JSON, CSV, YAML).
* **Data Preparation:** Splitting and filtering datasets to create input-ready files for downstream modeling.
* **NLP Based SQL Querry Agent:** The User can querry information stored in the Sqlite database easily using Natural Language which is then executed by the Agent.
  
## 🛠 Setup & Dependencies

To get started, follow these steps to set up your environment:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/NinadGawali/AI_Cricket_Analysis.git](https://github.com/NinadGawali/AI_Cricket_Analysis.git)
    cd AI_Cricket_Analysis
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate        # For macOS / Linux
    # venv\Scripts\activate         # For Windows
    ```

3.  **Install required packages:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
## ▶️ Usage / How to Run

    ```bash
    streamlit run app.py
    ```

## 🎥 Sample Video


https://github.com/user-attachments/assets/820120d3-01d9-4f82-950c-65f5b334b9fd


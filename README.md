# Tropical Cloud Cluster Identification Using INSAT Satellite Data

## Project Overview

This project develops an AI/ML-based algorithm to identify and classify tropical cloud clusters using half-hourly satellite data from the INSAT satellite series. The approach includes cloud mask generation, feature extraction, temporal alignment, cluster tracking, labeling, and model training to enable accurate and timely detection of tropical cloud clusters.

## Features

- Preprocessing of INSAT satellite imagery
- Cloud mask generation using KMeans clustering
- Feature extraction for cloud cluster characterization
- Temporal alignment and cluster tracking
- Supervised machine learning model for classification
- Model training, evaluation, and visualization

## Dataset

The dataset consists of half-hourly satellite images from INSAT (Indian National Satellite System). The raw satellite data and preprocessed data files are used for clustering and feature extraction.

*(Add any relevant links or instructions about how to access or download the data.)*

## Project Structure
```
project-root
│
├── data/ # Raw and processed satellite data files
├── modules/ # Python modules for different pipeline stages
│ ├── preprocessing.py # Functions for image preprocessing and cloud masking
│ ├── feature_extraction.py # Feature extraction from clustered data
│ ├── temporal_alignment.py # Temporal alignment and cluster tracking
│ ├── model_training.py # Model training and evaluation code
│ └── utils.py # Utility functions
├── notebooks/ # Jupyter notebooks for experimentation and visualization
├── main.py # Main script to run the full pipeline
├── requirements.txt # Python dependencies
└── README.md # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/insat-cloud-cluster.git
   cd insat-cloud-cluster
Create a Python virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
.\venv\Scripts\activate    # Windows
```
## Install dependencies:

```bash
pip install -r requirements.txt
```
## Usage
Run the main pipeline script to process data, train the model, and generate outputs:

```bash
python main.py
```
You can also explore the Jupyter notebooks in the notebooks/ folder for step-by-step analysis and visualization.

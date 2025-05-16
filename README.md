# A machine learning approach to breast cancer malignancy classification using nuclear morphometric features from Fine Needle Aspirate (FNA) cytology data

## Project Description

This project has the goal to develop and robustly evaluate various machine learning classifier models for the binary classification of breast cancer tumor cells as either malignant or benign.  The analysis utilizes the "breast_cancer.csv" dataset, which contains 512 tumor samples characterized by 30 numerical features derived from digitized images of fine needle aspirates (FNA) of breast masses.  The project pilars involve conducting exploratory data analysis, implementing a repeated Nested Cross-Validation (rnCV) pipeline to assess various classification algorithms, identifying the best-performing model, and training a final model instance on the entire dataset for potential deployment.

## Key Components

- **Exploratory Data Analysis**: Preprocessing of FNA data and exploration of the feature space
- **Model Training**: Implementation of various classification models
- **Model performance evaluation**: Performance assessment using metrics specialized for imbalanced classes
- **Overfitting Analysis**: Detection and visualization of model overfitting

## Project structure

    .
    ├── data/
    │   └── breast_cancer.csv
    ├── models/
    │   ├── winner_model.pkl                # The best performing model
    ├── notebooks/
    ├── src/                                # Source code for reusable functions and classes
    |   ├── _init_.py                       # Initiate src as a Python package
    │   ├── eda.py                          # Functions used for the exploratory data analysis (EDA)
    │   ├── ml_viz.py                       # Functions used for vizualization of the model results comparison
    │   ├── nested_cv.py                    # rnCV class with all relevant methods for training and testing
    ├── .gitignore
    ├── ex_2.yml            # Micromamba environment dependences to ensure reproducibility
    ├── README.md
    └── setup.py            # Package setup python script used for editable pip installation

## Installation

### 1. Prerequisites

**Git:** Used for version control

**Micromamba:** Used for setting a reproducible python virtual environment

**Jupyter Lab or Jupyter Notebook:** Used to run all the notebooks containing the analyses

### 2. Clone the Repository

Clone the assignment repository to your local machine and cd into it.

git clone [[Assignment-2](https://github.com/NeoCorteX33/Assignment-2)]

cd Assignment-2

### 3. Setup micromamba environment

Create environment and activate the micromamba environment by:

micromamba env create -f ex_2.yml

micromamba activate ex_2

### 4. Install the project as an editable package using pip

pip install -e .

## Usage guidlines

1. Launch Jupyter Lab or Jupyter Notebook from the project's root directory while your micromamba environment is activated.

2. Navigate to the notebooks/ directory.

3. Open the notebooks sequentially first EDA.ipynb and then ML_analysis.ipynb .

4. **Important:** Ensure that the notebooks you opened are using the ex_2 as the python kernel.

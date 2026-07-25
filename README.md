# Loan Default Prediction System

This is a GitHub-ready project demonstrating a **Loan Default Prediction System** built with Python, SQL-style feature engineering, and standard ML models.  
The synthetic dataset and code are designed to mirror real-world NBFC/banking analytics workflows (feature engineering, ETL-style transformations, model building, evaluation, and reporting).

**Included:**
- `data/loan_data.csv` - synthetic dataset (120k rows)
- `notebooks/model.ipynb` - walkthrough notebook
- `src/` - modular python scripts (preprocessing, features, training, evaluation)
- `dashboard.xlsx` - sample portfolio KPI dashboard (summary & cohorts)
- `requirements.txt` - libraries needed
- `Loan-Default-Prediction-System.zip` - ready-to-download archive

**Reference JD (user-provided):**
`/mnt/data/Vivriti Capital JD.pdf`

## How to run
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Explore `notebooks/model.ipynb` in JupyterLab / Jupyter Notebook.
3. Or run the scripts:
   ```bash
   python src/model_training.py --data data/loan_data.csv
   ```
4. Outputs (models, metrics) will be printed / saved to `outputs/`.

## Notes
- Dataset is synthetic for demonstration and interview purposes.
- The notebook contains exploratory analysis, feature engineering examples (SQL-like operations), model training with Logistic Regression and Random Forest, ROC-AUC and calibration checks, and a small segmentation/clustering section.

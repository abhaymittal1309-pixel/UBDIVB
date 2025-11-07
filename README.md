
# Streamlit Cloud App — Universal Bank Personal Loan Propensity

This app provides:
1. **Customer Insights**: 5 rich charts to guide conversion strategy.
2. **Modeling**: Train Decision Tree, Random Forest, and Gradient Boosting with one click. 
   - Metrics table (Train/Test Accuracy, Precision, Recall, F1, AUC)
   - ROC overlay
   - Train/Test confusion matrices
   - Feature importance
3. **Predict & Download**: Upload new data, score probabilities, set threshold, and download predictions.

## How to deploy on Streamlit Cloud
1. Create a new GitHub repo and upload these files (no folders needed).
2. On Streamlit Cloud, create a new app pointing to **app.py** as the main file.
3. Upload your `UniversalBank.csv` (or similar) in the app UI to start.

## Data requirements
Expected columns (case-insensitive, underscores accepted):
- ID (optional)
- Personal Loan (target; 0/1) — required for training
- Age, Experience, Income, Zip code, Family, CCAvg, Education, Mortgage
- Securities, CDAccount, Online, CreditCard (0/1)

The app auto-normalizes column names (spaces & dots to underscores) and fixes negative `Experience` to the median of non-negative values.

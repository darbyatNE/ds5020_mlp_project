import pandas as pd

# Read the original CSV file

dat = pd.read_csv("data/lin_reg_data/HousingData.csv")

# Select columns you want to keep
selected_columns = ["CRIM", "AGE", "DIS","MEDV"]

# Create a new DataFrame with only selected columns
new_data = dat[selected_columns]

# Save the new DataFrame to a new CSV file
new_data.to_csv("data/lin_reg_data/HousingData_Lmtd.csv", index=False)
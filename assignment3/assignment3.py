import pandas as pd
from datetime import datetime

sales = pd.read_csv("assignment3/AWSales.csv")
cust = pd.read_csv("assignment3/AWCustomers.csv")
df = pd.merge(sales, cust, on="CustomerID", how="inner")

df["BirthDate"] = pd.to_datetime(df["BirthDate"], errors="coerce")
df["Age"] = ((pd.to_datetime("today") - df["BirthDate"]).dt.days / 365.25).round()

selected = df[[
    "BikeBuyer", "AvgMonthSpend", "Age", "Education", "Occupation", 
    "Gender", "MaritalStatus", "HomeOwnerFlag", 
    "NumberCarsOwned", "NumberChildrenAtHome", "TotalChildren", 
    "YearlyIncome"
]]
print(selected)



variable_type = {
    "Discrete": [
        "BikeBuyer", "Gender", "MaritalStatus", "Education", "Occupation",
        "HomeOwnerFlag", "NumberCarsOwned", "NumberChildrenAtHome", "TotalChildren"
    ],
    "Continuous": [
        "Age", "YearlyIncome", "AvgMonthSpend"
    ]
}


measurement_scale = {
    "Nominal": [
        "BikeBuyer", "Gender", "MaritalStatus", "Occupation", "HomeOwnerFlag"
    ],
    "Ordinal": [
        "Education"
    ],
    "Ratio": [
        "Age", "YearlyIncome", "AvgMonthSpend", 
        "NumberCarsOwned", "NumberChildrenAtHome", "TotalChildren"
    ]
}

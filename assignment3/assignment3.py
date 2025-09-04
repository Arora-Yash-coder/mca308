import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency

sales = pd.read_csv("assignment3/AWSales.csv")
cust = pd.read_csv("assignment3/AWCustomers.csv")
df = pd.merge(sales, cust, on="CustomerID", how="inner")

df["BirthDate"] = pd.to_datetime(df["BirthDate"], errors="coerce")
df["Age"] = ((pd.to_datetime("today") - df["BirthDate"]).dt.days / 365.25).round()

selected = df[[
    "BikeBuyer", "AvgMonthSpend", "Age", "Education", "Occupation", 
    "Gender", "MaritalStatus", "HomeOwnerFlag", 
    "NumberCarsOwned", "NumberChildrenAtHome", "TotalChildren", 
    "YearlyIncome", "CommuteDistance"
]]
print(selected.head())

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
        "Education", "CommuteDistance"
    ],
    "Ratio": [
        "Age", "YearlyIncome", "AvgMonthSpend", 
        "NumberCarsOwned", "NumberChildrenAtHome", "TotalChildren"
    ]
}

df["YearlyIncome"] = pd.to_numeric(df["YearlyIncome"], errors="coerce")

buyers = df[df["BikeBuyer"] == 1]["YearlyIncome"].dropna()
non_buyers = df[df["BikeBuyer"] == 0]["YearlyIncome"].dropna()
mean_buyers, mean_nonbuyers = buyers.mean(), non_buyers.mean()
print("Average Income of Buyers:", round(mean_buyers,2))
print("Average Income of Non-Buyers:", round(mean_nonbuyers,2))
t_stat, p_val = ttest_ind(buyers, non_buyers)
print("T-test p-value:", p_val)

plt.figure(figsize=(6,4))
sns.barplot(x=["Non-Buyers","Buyers"], y=[mean_nonbuyers, mean_buyers], palette="Set2")
plt.title("Average Yearly Income: Buyers vs Non-Buyers")
plt.ylabel("Average Yearly Income")
plt.show()

commute_stats = df.groupby("CommuteDistance")["BikeBuyer"].mean() * 100
print("\nPercentage of customers buying bikes by commute distance:\n", commute_stats)

plt.figure(figsize=(8,5))
commute_stats.plot(kind="bar", color="skyblue")
plt.title("Bike Buyers by Commute Distance (%)")
plt.ylabel("Percentage")
plt.show()

plt.figure(figsize=(8,5))
sns.kdeplot(df[df["BikeBuyer"]==1]["Age"], label="Buyers", shade=True)
sns.kdeplot(df[df["BikeBuyer"]==0]["Age"], label="Non-Buyers", shade=True)
plt.title("Age Distribution: Buyers vs Non-Buyers")
plt.xlabel("Age")
plt.legend()
plt.show()

marital_ct = pd.crosstab(df["MaritalStatus"], df["BikeBuyer"])
print("\nBike Buyer counts by Marital Status:\n", marital_ct)
chi2, p, dof, exp = chi2_contingency(marital_ct)
print("Chi-Square p-value:", p)

marital_ct.plot(kind="bar", stacked=True, figsize=(8,5), colormap="Accent")
plt.title("Bike Buyers by Marital Status")
plt.ylabel("Count")
plt.show()

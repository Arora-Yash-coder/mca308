import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.datasets import load_iris

data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9,10],
    'refund': ['Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No'],
    'income': [125, 100, 70, 120, 95, 60, 220, 85, 75, 90],
    'status': ['Single', 'Married', 'Single', 'Married', 'Divorced', 'Married', 'Divorced', 'Single', 'Married', 'Single'],
    'Cheat' : ['No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)
print("Q1 Dataset:\n", df)

print("\nQ2 Selected Rows:\n", df.loc[[0, 4, 7, 8]])

df.to_csv("assignment2/data_assignment2.csv", index=False)
df_csv = pd.read_csv("assignment2/data_assignment2.csv")
print("\nQ3 CSV File:\n", df_csv)

dept_type = CategoricalDtype(categories=['Science', 'Commerce', 'Arts'], ordered=False) 
passfail_type = CategoricalDtype(categories=['Pass', 'Fail'], ordered=False)  
grade_type = CategoricalDtype(categories=['Low', 'Medium', 'High'], ordered=True) 


student_data = {
    'Roll_No': pd.Series([101, 102, 103, 104, 105], dtype='int'),
    'Name': pd.Series(['Anubhav', 'Guppi', 'Sumit', 'Asir', 'Edward']), 
    'Department': pd.Series(['Science', 'Commerce', 'Arts', 'Science', 'Commerce'], dtype=dept_type),
    'Pass_Fail': pd.Series(['Pass', 'Pass', 'Fail', 'Pass', 'Fail'], dtype=passfail_type),  
    'Grade_Level': pd.Series(['High', 'Medium', 'Low', 'High', 'Medium'], dtype=grade_type),  
    'Marks': pd.Series([85.5, 78.0, 56.5, 92.0, 67.5], dtype='float')
}

students_df = pd.DataFrame(student_data)
print("\nQ4.\n",students_df)
print("\nQ4. Dtypes:\n", students_df.dtypes)


print("\nQ5.1 Rows 3 to 7:\n", df.iloc[3:8])
print("\nQ5.2 Rows 4 to 8, Cols 2 to 4:\n", df.iloc[4:9, 2:5])
print("\nQ5.3 All Rows, Cols 1 to 3:\n", df.iloc[:, 1:4])

iris = load_iris(as_frame=True)
iris_df = iris.frame
print("\nQ6 First 5 rows of Iris:\n", iris_df.head())

iris_mod = iris_df.drop(index=4).drop(iris_df.columns[3], axis=1)
print("\nQ7 Modified Iris:\n", iris_mod.head())

emp_data = {
    'Employee_ID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward'],
    'Department': ['HR', 'IT', 'IT', 'Marketing', 'Sales'],
    'Age': [29, 34, 41, 28, 38],
    'Salary': [50000, 70000, 65000, 55000, 60000],
    'Years_of_Experience': [4, 8, 10, 3, 12],
    'Joining_Date': ['2020-03-15', '2017-07-19', '2013-06-01', '2021-02-10', '2010-11-25'],
    'Gender': ['Female', 'Male', 'Male', 'Female', 'Male'],
    'Bonus': [5000, 7000, 6000, 4500, 5000],
    'Rating': [4.5, 4.0, 3.8, 4.7, 3.5]
}
emp_df = pd.DataFrame(emp_data)

print("\nQ8.a Shape:", emp_df.shape)
print("\nQ8.b Summary:\n")
print(emp_df.info())
print("\nQ8.c Stats:\n", emp_df.describe())
print("\nQ8.d First 5:\n", emp_df.head())
print("\nQ8.d Last 3:\n", emp_df.tail(3))
print("\nQ8.e.i Avg Salary:", emp_df['Salary'].mean())
print("Q8.e.ii Total Bonus:", emp_df['Bonus'].sum())
print("Q8.e.iii Youngest Age:", emp_df['Age'].min())
print("Q8.e.iv Highest Rating:", emp_df['Rating'].max())
print("\nQ8.f Sorted:\n", emp_df.sort_values(by='Salary', ascending=False))

def categorize(r):
    if r >= 4.5:
        return "Excellent"
    elif r >= 4.0:
        return "Good"
    else:
        return "Average"

emp_df['Performance'] = emp_df['Rating'].apply(categorize)
print("\nQ8.g With Performance:\n", emp_df)

print("\nQ8.h Missing:\n", emp_df.isnull().sum())

emp_df.rename(columns={'Employee_ID': 'ID'}, inplace=True)
print("\nQ8.i Renamed:\n", emp_df.head())

print("\nQ8.j.i >5 years exp:\n", emp_df[emp_df['Years_of_Experience'] > 5])
print("\nQ8.j.ii IT Dept:\n", emp_df[emp_df['Department'] == 'IT'])

emp_df['Tax'] = emp_df['Salary'] * 0.10
print("\nQ8.k With Tax:\n", emp_df)

emp_df.to_csv("assignment2/employees.csv", index=False)
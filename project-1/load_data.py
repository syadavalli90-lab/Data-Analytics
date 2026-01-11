import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
# importing new libraries

sales_path = Path("sales.csv")
features_path = Path("features.csv")
stores_path = Path("stores.csv")
# creating paths to all three csv files

sales = pd.read_csv(sales_path)
features = pd.read_csv(features_path)
stores = pd.read_csv(stores_path)
# using "pd.read" so I can read the .csv files

print("Sales shape: ", sales.shape)
print("Features shape: ", features.shape)
print("Stores shape: ", stores.shape)
# prints the numbers of rows & columns in each file

print("\nSales Preview: ")
print(sales.head())
# prints the first 5 rows of the sales file

df = sales.merge(features, on=["Store", "Date"], how="left")
print("\nAfter joining sales + features:") 
print("Shape:", df.shape) 
print(df.head())
# merges the sales and features files together

df = df.merge(stores, on="Store", how="left")
print("\nAfter joining stores:")
print(df.head())
# merges the df file with the stores file

print("\nMissing values check:")
print(df.isna().sum().head(10))
# checks for missing values in the df file

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
print(df["Date"].dtype)
print(df["Date"].head())
# converts the "Date" column to datetime format

print(df["Date"].head())
print(df["Date"].dtype)
# prints the first 5 rows of the "Date" column and its data type

df["Year"] = df["Date"].dt.year
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
# creates new columns "Year" and "Week" from the "Date" column

print(df[["Date", "Year", "Week"]].head(10))
print(df["Week"].min(), df["Week"].max())
# prints the first 10 rows of the "Date", "Year", and "Week" columns and the minimum and maximum values of the "Week" column

dept_sales = (
    df.groupby(["Store", "Dept", "Year", "Week"], as_index=False)["Weekly_Sales"]
      .sum()
)
print(dept_sales.head())
print(dept_sales.shape)
# groups the df file by "Store", "Dept", "Year", and "Week" and sums the "Weekly_Sales" column

df["Year"] = df["Date"].dt.year
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
# creates new columns "Week" & "Year" to be used for queries

print(df[["Date", "Year", "Week"]].head(10))
print(df["Week"].min(), df["Week"].max())
# prints the first 10 rows of the "Date", "Year", and "Week" columns and the minimum and maximum values of the "Week" column

dept_sales = (
    df.groupby(["Store", "Dept", "Year", "Week"], as_index=False)["Weekly_Sales"]
      .sum()
)
print(dept_sales.head())
print(dept_sales.shape)
# groups the df file by "Store", "Dept", "Year", and "Week" and sums the "Weekly_Sales" column

forecast = (
    dept_sales
    .groupby(["Store", "Dept", "Week"], as_index=False)["Weekly_Sales"]
    .mean()
    .rename(columns={"Weekly_Sales": "Predicted_Weekly_Sales"})
)
print(forecast.head())
print(forecast["Predicted_Weekly_Sales"].describe())
# groups the dept_sales file by "Store", "Dept", and "Week" and calculates the mean of the "Weekly_Sales" column, renaming it to "Predicted_Weekly_Sales"

next_year = df["Year"].max() + 1
forecast["Predicted_Year"] = next_year
# creates a new column "Predicted_Year" with the value of the maximum year in the "Year" column plus 1

print(forecast["Predicted_Year"].unique())
print("Stores:", forecast["Store"].nunique())
print("Departments:", forecast["Dept"].nunique())
print("Weeks:", forecast["Week"].min(), "to", forecast["Week"].max())
# prints the unique values of the "Predicted_Year" column, the number of unique stores, the number of unique departments, and the minimum and maximum values of the "Week" column

store_id = 1
dept_id = 1

hist = dept_sales[
    (dept_sales["Store"] == store_id) &
    (dept_sales["Dept"] == dept_id)
]

pred = forecast[
    (forecast["Store"] == store_id) &
    (forecast["Dept"] == dept_id)
]

sns.lineplot(data=hist, x="Week", y="Weekly_Sales", hue="Year")
sns.lineplot(
    data=pred,
    x="Week",
    y="Predicted_Weekly_Sales",
    color="black",
    linewidth=3,
    label="Forecast"
)

plt.title(f"Weekly Sales Forecast — Store {store_id}, Dept {dept_id}")
plt.show()
# creating plots for the sales data

# End of Task 1 (Predict the department-wide sales for each store for the following year)

# Start of Task 2 (Model the effects of markdowns on holiday weeks)

print(df.columns)
markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"] 

df["Any_Markdown"] = df[markdown_cols].notna().any(axis=1)
print(df["Any_Markdown"].value_counts(dropna=False))
# creates a new column "Any_Markdown" that indicates whether any of the markdown columns have a non-null value

print(df.columns.tolist())
print([col for col in df.columns if "holiday" in col.lower()])
df = df.rename(columns={"IsHoliday_x": "IsHoliday"}).drop(columns=["IsHoliday_y"])
print([c for c in df.columns if "IsHoliday" in c])
print(df["IsHoliday"].value_counts(dropna=False))
# renames the "IsHoliday_x" column to "IsHoliday" and drops the "IsHoliday_y" column

holiday_df = df[df["IsHoliday"] == True].copy()

print("Holiday rows:", holiday_df.shape[0])
print(holiday_df["IsHoliday"].value_counts(dropna=False))
print(holiday_df["Any_Markdown"].value_counts(dropna=False))
# creates a new dataframe "holiday_df" that contains only the rows where "IsHoliday" is True

holiday_summary = (
    holiday_df
    .groupby("Any_Markdown")["Weekly_Sales"]
    .agg(count="count", mean="mean", median="median")
    .reset_index()
)
print(holiday_summary)
md_mean = holiday_summary.loc[holiday_summary["Any_Markdown"] == True, "mean"].iloc[0]
no_md_mean = holiday_summary.loc[holiday_summary["Any_Markdown"] == False, "mean"].iloc[0]
pct_change = (md_mean - no_md_mean) / no_md_mean * 100
print(f"Avg holiday sales WITH markdown: {md_mean:,.2f}")
print(f"Avg holiday sales WITHOUT markdown: {no_md_mean:,.2f}")
print(f"Percent difference: {pct_change:.2f}%")
# calculates the average holiday sales with and without markdowns, and the percentage difference

sns.barplot(data=holiday_df, x="Any_Markdown", y="Weekly_Sales")
plt.title("Holiday Weekly Sales: Markdown vs No Markdown")
plt.xlabel("Any Markdown?")
plt.ylabel("Weekly Sales")
plt.show()
# creates plot for the effect of markdowns on holiday weeks

# End of Task 2 (Model the effects of markdowns on holiday weeks)

# Start of Task 3 (Provide recommended actions based on the insights drawn, with prioritization placed on largest business impact)

print("What are the insights we gained from the data?\n1 - sales show strong weekly seasonality, meaning that certain weeks peak in sales")
print("2 - markdowns are causing a 5.63% increase in average weekly sales in holiday weeks")
print("\nMy recommendations are as follows:\n1 - Increase inventory and staffing ahead of peak forecasted weeks & reduce overstock in low-forecast weeks")
print("This will create fewer stockouts during high-demand periods, lowers carrying costs during slow weeks, & creates direct margin improvement")
print("2 - maintain or expand markdown strategy during holiday weeks & focus markdown budgets on historically responsive periods")
print("This will create higher revenue during peak demand & better ROI on promotional spend")

# End of Task 3

# End of Project
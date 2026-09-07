# DataVision-Sales-Data-Cleaning-Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. CREATE / LOAD DATASET
# --------------------------------------------------

# If you have a CSV file, use:
# df = pd.read_csv("sales_data.csv")

# Creating a sample raw dataset
data = {
    "Order_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                 111, 112, 113, 114, 115],
    "Product": ["Laptop", "Mobile", "Tablet", "Laptop", "Mobile",
                "Headphones", "Tablet", "Laptop", "Mobile", "Keyboard",
                "Laptop", "Mobile", "Tablet", "Headphones", "Laptop"],
    "Category": ["Electronics", "Electronics", "Electronics", "Electronics",
                 "Electronics", "Accessories", "Electronics", "Electronics",
                 "Electronics", "Accessories", "Electronics", "Electronics",
                 "Electronics", "Accessories", "Electronics"],
    "Quantity": [2, 3, 1, 2, np.nan, 4, 2, 1, 5, 3,
                 2, 4, 1, 3, 10],
    "Unit_Price": [50000, 20000, 30000, 50000, 20000, 2000, 30000,
                   50000, 20000, 1500, 50000, 20000, 30000, 2000, 50000],
    "Region": ["South", "North", "East", "West", "South",
               "North", "East", "South", "West", "North",
               "South", "East", "West", "South", "South"],
    "Payment_Mode": ["UPI", "Card", "Cash", "UPI", "Card",
                     "UPI", "Cash", "Card", "UPI", "Cash",
                     "UPI", "Card", "UPI", "Cash", "Card"]
}

df = pd.DataFrame(data)

print("\n========== RAW DATA ==========")
print(df)

# --------------------------------------------------
# 2. BASIC INFORMATION
# --------------------------------------------------

print("\n========== DATA INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# --------------------------------------------------
# 3. CHECK MISSING VALUES
# --------------------------------------------------

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Fill missing Quantity with median
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())

# --------------------------------------------------
# 4. CHECK AND REMOVE DUPLICATES
# --------------------------------------------------

print("\n========== DUPLICATES ==========")
print("Number of duplicates:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# --------------------------------------------------
# 5. CREATE TOTAL SALES COLUMN
# --------------------------------------------------

df["Total_Sales"] = df["Quantity"] * df["Unit_Price"]

print("\n========== CLEANED DATA ==========")
print(df)

# --------------------------------------------------
# 6. OUTLIER DETECTION USING IQR
# --------------------------------------------------

Q1 = df["Total_Sales"].quantile(0.25)
Q3 = df["Total_Sales"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\n========== OUTLIER INFORMATION ==========")
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

outliers = df[
    (df["Total_Sales"] < lower_limit) |
    (df["Total_Sales"] > upper_limit)
]

print("\nOutliers:")
print(outliers)

# Remove outliers
df_clean = df[
    (df["Total_Sales"] >= lower_limit) &
    (df["Total_Sales"] <= upper_limit)
]

# --------------------------------------------------
# 7. SALES ANALYSIS
# --------------------------------------------------

total_sales = df_clean["Total_Sales"].sum()
average_sales = df_clean["Total_Sales"].mean()
total_orders = len(df_clean)

print("\n========== SALES SUMMARY ==========")
print("Total Sales     :", total_sales)
print("Average Sales   :", round(average_sales, 2))
print("Total Orders    :", total_orders)

# --------------------------------------------------
# 8. SALES BY PRODUCT
# --------------------------------------------------

product_sales = df_clean.groupby("Product")["Total_Sales"].sum()

print("\n========== SALES BY PRODUCT ==========")
print(product_sales)

# --------------------------------------------------
# 9. SALES BY CATEGORY
# --------------------------------------------------

category_sales = df_clean.groupby("Category")["Total_Sales"].sum()

print("\n========== SALES BY CATEGORY ==========")
print(category_sales)

# --------------------------------------------------
# 10. SALES BY REGION
# --------------------------------------------------

region_sales = df_clean.groupby("Region")["Total_Sales"].sum()

print("\n========== SALES BY REGION ==========")
print(region_sales)

# --------------------------------------------------
# 11. PAYMENT MODE ANALYSIS
# --------------------------------------------------

payment_sales = df_clean.groupby("Payment_Mode")["Total_Sales"].sum()

print("\n========== SALES BY PAYMENT MODE ==========")
print(payment_sales)

# --------------------------------------------------
# 12. VISUALIZATION SETTINGS
# --------------------------------------------------

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# 13. BAR CHART - SALES BY PRODUCT
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.barplot(
    x=product_sales.index,
    y=product_sales.values
)

plt.title("Total Sales by Product")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 14. BAR CHART - SALES BY REGION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    x=region_sales.index,
    y=region_sales.values
)

plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 15. PIE CHART - PAYMENT MODE
# --------------------------------------------------

plt.figure(figsize=(7, 7))

plt.pie(
    payment_sales.values,
    labels=payment_sales.index,
    autopct="%1.1f%%"
)

plt.title("Sales Distribution by Payment Mode")
plt.show()

# --------------------------------------------------
# 16. HISTOGRAM - SALES DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df_clean["Total_Sales"],
    bins=10,
    kde=True
)

plt.title("Sales Distribution")
plt.xlabel("Total Sales")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 17. BOX PLOT - OUTLIER VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["Total_Sales"]
)

plt.title("Sales Outlier Detection")
plt.xlabel("Total Sales")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 18. TOP PRODUCTS
# --------------------------------------------------

top_products = product_sales.sort_values(
    ascending=False
).head(5)

print("\n========== TOP 5 PRODUCTS ==========")
print(top_products)

# --------------------------------------------------
# 19. SAVE CLEANED DATA
# --------------------------------------------------

df_clean.to_csv(
    "cleaned_sales_data.csv",
    index=False
)

print("\nCleaned dataset saved as: cleaned_sales_data.csv")

print("\n========== PROJECT COMPLETED ==========")

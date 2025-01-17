import pandas as pd
import matplotlib.pyplot as plt

#Loading the dataset
df = pd.read_csv('https://storage.googleapis.com/courses_data/Assignment%20CSV/finance_liquor_sales.csv')

#Data preprocessing
#------------------
#Keeping only the rows whose date value lies between 2016 and 2019
df["date"] = pd.to_datetime(df["date"])
filtered_df = df[(df["date"].dt.year >= 2016) & (df["date"].dt.year <= 2019)].reset_index()
#Determining how many missing values each of our columns has
filtered_df.info()

#Task A
#------
#Discerning the most popular item in each zipcode
#------------------------------------------------
#Grouping the data by the first two columns and then measuring the popularity based on the sum of bottles_sold for each zip-item pair.
bottles_sold = filtered_df.groupby(["zip_code", "item_number"])["bottles_sold"].sum().reset_index()
#Converting zip_code column into an integer
bottles_sold["zip_code"] = bottles_sold["zip_code"].astype(int)
#For each zip_code, we find the index of the item_number that has the most bottles sold
idx = bottles_sold.groupby("zip_code")["bottles_sold"].idxmax()
max_bottles = bottles_sold.loc[idx].reset_index()
#Sorting the values in descending order
sorted_values = max_bottles.sort_values(by="bottles_sold", ascending=False)
#Using the matplotlib module to create a convenient plot
plt.figure(figsize=(10,6))
plt.bar(sorted_values["zip_code"].astype(str), sorted_values["bottles_sold"], color="skyblue")
plt.xlabel("Zip Code")
plt.ylabel("Bottles Sold")
plt.title("Max bottles sold per zip code")
plt.xticks(rotation=45)
plt.show()

#Task B
#------
#Computing the sales percentage per store(in dollar)
#---------------------------------------------------
#Finding the total amount of sales
total_sales = sum(filtered_df["sale_dollars"])
#Calculating the dollar amount of sales for each store
sales = filtered_df.groupby("store_name")["sale_dollars"].sum()
#Computing the percentage of each store's sales
percentage_sales = (sales * 100 / total_sales)
#Sorting the values in ascending order
percentage_sorted_sales = percentage_sales.sort_values(ascending=True)
#Using the matplotlib module and pandas module to create a plot
p = plt.barh(percentage_sorted_sales.index, percentage_sorted_sales.values, height=0.7)
plt.title("%Sales by store")
plt.xlabel("%Sales", fontsize=12)
plt.bar_label(p, fmt="%.2f")
plt.xlim([0,20])
plt.show()

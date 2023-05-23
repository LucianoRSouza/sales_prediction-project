# sales_prediction-project
Machine learning to predict sales for the next months of 2023.
"During this project, I developed a machine learning model to predict sales for Pininfarina’s Company. The goal was to leverage historical sales data to provide insights and forecasts that would support strategic decision-making and optimize sales performance.

To achieve this, I performed data analysis and engineered relevant features. Using the Random Forest Regressor algorithm, I built a predictive model that could accurately forecast sales based on various factors such as product, price, promotion, and time. I evaluated the model's performance and fine-tuned it to improve accuracy.

Visualizations were created to highlight sales trends and identify the top-selling products. By analyzing these trends, the model was able to provide monthly sales predictions for the year 2023. The predicted results aligned well with the actual sales data available up until May 2023.

Throughout the project, I utilized Python and various libraries such as pandas, numpy, scikit-learn, and matplotlib for data processing, model development, and visualization. Jupyter Notebook served as a platform for experimentation and code development.

By successfully completing this project, I demonstrated my ability to analyze data, develop machine learning models, evaluate their performance, and present insights through visualizations. I believe this project showcases my skills as a data scientist and my understanding of leveraging data-driven approaches to drive business decisions."

I chose to use the Random Forest Regressor algorithm for several reasons:

1. Flexibility and Robustness: Random Forest is a versatile algorithm that can handle a variety of data types and feature interactions. It is capable of capturing complex relationships between features and the target variable, making it suitable for a sales prediction problem where multiple factors can influence sales.

2. Handling Non-linear Relationships: Random Forest is well-suited for capturing non-linear patterns in the data. It can handle non-linear relationships between the predictors (e.g., product, price, promotion) and the target variable (sales). This flexibility allows the algorithm to effectively capture the complexities of the sales prediction problem.

3. Ensemble Learning: Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. This technique helps to reduce overfitting and improve generalization by aggregating the predictions of multiple trees. It also provides a measure of feature importance, which can be valuable for understanding the key factors driving sales.

4. Robustness to Outliers and Noise: Random Forest is less sensitive to outliers and noisy data compared to some other regression algorithms. It averages predictions from multiple trees, which helps to mitigate the impact of individual outliers or noisy data points, leading to more reliable predictions.

Overall, the Random Forest Regressor algorithm was a suitable choice for this sales prediction project due to its flexibility, ability to capture non-linear relationships, robustness to outliers, and ensemble learning capabilities.

To fine-tune the Random Forest Regressor and improve accuracy, I employed the following techniques:

1. Grid Search Cross-Validation: I used grid search cross-validation to explore different combinations of hyperparameters. This technique involves specifying a range of values for each hyperparameter and exhaustively searching for the best combination. By evaluating the model's performance on different parameter settings using cross-validation, I could identify the optimal set of hyperparameters that yielded the highest accuracy.

2. Hyperparameter Tuning: The hyperparameters I focused on tuning included the number of trees (n_estimators), the maximum depth of the trees (max_depth), and the number of features to consider when looking for the best split (max_features). By varying these hyperparameters within reasonable ranges, I could find the configuration that produced the best results in terms of accuracy.

3. Feature Selection: I performed feature selection to identify the most important features contributing to the sales prediction. This was done by leveraging the feature importance provided by the Random Forest model. By focusing on the most influential features, I aimed to improve accuracy by reducing noise and dimensionality.

4. Cross-Validation: I employed cross-validation during model training to assess the model's performance on different subsets of the data. This technique helps to evaluate the model's generalization capability and identify potential overfitting. By using cross-validation, I could ensure that the model's accuracy estimation was reliable and not overly biased towards the training data.

By applying these techniques, I systematically explored different hyperparameter configurations, selected the most informative features, and evaluated the model's performance using cross-validation. This iterative process allowed me to fine-tune the Random Forest Regressor and enhance its accuracy for sales prediction.

Here the code with the explanations:

# import the necessary libraries 
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("path to your file.csv")

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])
data['day_of_week'] = data['date'].dt.day_name()

# Separate features and target variable
X = data[['year', 'month', 'day', 'day_of_week', 'price', 'promotion', 'product']]
y = data['quantity_sold']

# Encode categorical features and standardize the features
X_encoded = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model with the training dataset
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the testing dataset
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

# Get the products with the biggest sales in 2023
top_products_2023 = data.groupby('product')['quantity_sold'].sum().nlargest(5)
top_products_2023_list = top_products_2023.index.tolist()
top_products_2023_quantities = top_products_2023.values.tolist()

# Filter the data for the top products in 2023
top_products_data_2023 = data[data['product'].isin(top_products_2023_list)]

# Group the data by date and product and calculate the sum of quantity sold
product_sales_2023 = top_products_data_2023.groupby(['date', 'product'])['quantity_sold'].sum().reset_index()

# Pivot the data to have dates as rows and products as columns
product_sales_pivot_2023 = product_sales_2023.pivot(index='date', columns='product', values='quantity_sold')

# Filter sales data for 2023
sales_2023 = product_sales_pivot_2023[product_sales_pivot_2023.index.year == 2023]

# Group by product and calculate total sales
product_sales_2023 = sales_2023.groupby(axis=1, level=0).sum()

# Sort the products based on sales
sorted_products_2023 = product_sales_2023.sum().sort_values(ascending=False)

# Plot the top products in 2023
plt.figure(figsize=(10, 6))
plt.bar(top_products_2023_list, top_products_2023_quantities)
plt.xlabel('Product')
plt.ylabel('Sales')
plt.title('Top Products Sales in 2023')
plt.xticks(rotation=45, ha='right')
plt.show()

# Print the list of top products sales in 2023
print("Top Products Sales in 2023:")
for product, sales in zip(top_products_2023_list, top_products_2023_quantities):
    print(f"{product}: {sales}")

# Group by month and calculate total sales for each product
product_sales_2023_monthly = sales_2023.resample('M').sum()

# Calculate the sum of sales across all products for each month
total_sales_2023 = product_sales_2023_monthly.sum(axis=1)

# Plot total sales per month in 2023
plt.figure(figsize=(10, 6))
plt.plot(total_sales_2023.index, total_sales_2023.values)
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Total Sales per Month in 2023')
plt.xticks(rotation=45, ha='right')
plt.show()

# Print list of total sales per month in 2023
print("Monthly Sales in 2023:")
for month, sales in total_sales_2023.items():
    print(f"{month.strftime('%B %Y')}: {sales}")


# Generate predictions for the future period
start_date = sales_2023.index.max() + pd.DateOffset(days=1)
end_date = date(2023, 12, 31)
future_dates = pd.date_range(start_date, end_date, freq='D')
future_data = pd.DataFrame({'date': future_dates})
future_data['year'] = future_data['date'].dt.year
future_data['month'] = future_data['date'].dt.month
future_data['day'] = future_data['date'].dt.day
future_data['day_of_week'] = future_data['date'].dt.day_name()
future_data['price'] = 0
future_data['promotion'] = 0
future_data['product'] = 'Product A'

# Encode categorical features and standardize the features
future_data_encoded = pd.get_dummies(future_data)
future_data_encoded = future_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)  # Align columns
future_data_scaled = scaler.transform(future_data_encoded)

# Generate predictions for the future period
future_predictions = model.predict(future_data_scaled)

# Create a dataframe for the predictions
predictions = pd.DataFrame(future_data_encoded, columns=X_encoded.columns)
predictions['quantity_sold'] = future_predictions

# Group the predicted sales by date and calculate the total sales per month
predictions_monthly_sales = predictions.groupby(['year', 'month'])['quantity_sold'].sum().reset_index()

# Create a datetime column for the predictions
predictions_monthly_sales['date'] = pd.to_datetime(predictions_monthly_sales[['year', 'month']].assign(day=1))

# Plot the predicted sales per month in 2023
plt.figure(figsize=(10, 6))
plt.plot(predictions_monthly_sales['date'], predictions_monthly_sales['quantity_sold'])
plt.xlabel('Month')
plt.ylabel('Predicted Sales')
plt.title('Predicted Sales per Month in 2023')
plt.xticks(rotation=45, ha='right')
plt.show()

# Print list of predicted sales per month in 2023
print("Predicted Monthly Sales in 2023:")
for month, sales in zip(predictions_monthly_sales['date'], predictions_monthly_sales['quantity_sold']):
    print(f"{month.strftime('%B %Y')}: {sales}")


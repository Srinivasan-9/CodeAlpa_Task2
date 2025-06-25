
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from fpdf import FPDF
import re
import os

# Step 1: Prepare dataset
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
    'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
}
df = pd.DataFrame(data)

# Step 2: Correlation heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("heatmap.png")
plt.close()

# Step 3: TV vs Sales scatter plot
plt.figure(figsize=(5, 4))
plt.scatter(df['TV'], df['Sales'])
plt.title('TV Spend vs Sales')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.tight_layout()
plt.savefig("tv_vs_sales.png")
plt.close()

# Step 4: Train model
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Step 5: Actual vs Predicted Plot
plt.figure(figsize=(5, 4))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.tight_layout()
plt.savefig("actual_vs_pred.png")
plt.close()

# Step 6: ASCII clean function
def clean(text):
    return re.sub(r'[^\x00-\x7F]+',' ', text)

# Step 7: Create PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, clean("Sales Prediction Using Python"), ln=True, align='C')
pdf.set_font("Arial", '', 12)
pdf.cell(0, 10, "Date: June 24, 2025", ln=True)
pdf.cell(0, 10, "Presented by: Srinivasan G - Code Alpha Task 4", ln=True)

pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Introduction", ln=True)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 10, clean("This project uses Linear Regression to predict sales based on advertising spend across TV, Radio, and Newspaper."))

pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Dataset", ln=True)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 10, clean("Features:\n- TV\n- Radio\n- Newspaper\nTarget:\n- Sales (units sold)"))

pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Correlation Heatmap", ln=True)
pdf.image("heatmap.png", w=150)

pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "TV Spend vs Sales", ln=True)
pdf.image("tv_vs_sales.png", w=150)

pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Model Training & Prediction", ln=True)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 10, clean(f"Model trained using Linear Regression.\nR2 Score: {r2:.2f}"))

pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Actual vs Predicted Sales", ln=True)
pdf.image("actual_vs_pred.png", w=150)

pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Conclusion", ln=True)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 10, clean(f"TV and Radio are strong predictors of sales.\nNewspaper has weaker impact.\nModel R2 score: {r2:.2f}"))

pdf.output("Sales_Prediction_Report.pdf")
print("✅ Report saved as Sales_Prediction_Report.pdf")

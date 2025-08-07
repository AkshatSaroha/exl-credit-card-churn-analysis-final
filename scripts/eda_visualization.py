import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load local data
df = pd.read_csv('../data/processed/churn_cleaned.csv')

# Age distribution
sns.histplot(df['Age'], kde=True, color='green')
plt.title("Age Distribution")
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig("../data/processed/age_distribution.png")
plt.clf()

# Tenure distribution
sns.histplot(df['Tenure'], bins=10, color='purple')
plt.title('Tenure Distribution')
plt.xlabel('Tenure (Years)')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig("../data/processed/tenure_distribution.png")
plt.clf()


# Chruned vs non-churned
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x='Churn', palette='Set1')
plt.title('Churn vs Non-Churn Count')
plt.xticks([0, 1], ['Non-Churned', 'Churned'])
plt.xlabel('Customer Status')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig("../data/processed/churnVsNonChurn.png")
plt.clf()

# Correlation matrix
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("../data/processed/correlation_matrix.png")
plt.clf()

print("EDA visualizations saved in /data/processed/")

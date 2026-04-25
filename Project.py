import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

df = pd.read_csv("data.csv")

print(df.head())
print(df.info())
print(df.describe())

df.columns = df.columns.str.strip().str.upper()

df = df.drop_duplicates()

df['NO_OF_PAGES'] = pd.to_numeric(df['NO_OF_PAGES'], errors='coerce')
df['NO_OF_CLAIMS'] = pd.to_numeric(df['NO_OF_CLAIMS'], errors='coerce')

df['NO_OF_PAGES'] = df['NO_OF_PAGES'].fillna(df['NO_OF_PAGES'].mean())
df['NO_OF_CLAIMS'] = df['NO_OF_CLAIMS'].fillna(df['NO_OF_CLAIMS'].mean())

plt.figure()
plt.hist(df['NO_OF_PAGES'])
plt.title("Distribution of Number of Pages")
plt.xlabel("Number of Pages")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.boxplot(df['NO_OF_CLAIMS'])
plt.title("Boxplot of Number of Claims")
plt.xlabel("Claims")
plt.show()

plt.figure()
plt.scatter(df['NO_OF_PAGES'], df['NO_OF_CLAIMS'])
plt.title("Pages vs Claims")
plt.xlabel("Number of Pages")
plt.ylabel("Number of Claims")
plt.show()

plt.figure()
plt.plot(df['NO_OF_PAGES'])
plt.title("Trend of Number of Pages")
plt.xlabel("Index")
plt.ylabel("Number of Pages")
plt.show()

corr = df[['NO_OF_PAGES', 'NO_OF_CLAIMS']].corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks([0,1], ['PAGES','CLAIMS'])
plt.yticks([0,1], ['PAGES','CLAIMS'])
plt.title("Correlation Heatmap")
plt.show()


X = df[['NO_OF_PAGES']]
y = df['NO_OF_CLAIMS']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure()
plt.scatter(df['NO_OF_PAGES'], df['NO_OF_CLAIMS'])
plt.plot(df['NO_OF_PAGES'], y_pred)
plt.title("Linear Regression")
plt.xlabel("Number of Pages")
plt.ylabel("Number of Claims")
plt.show()

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

sample = df['NO_OF_PAGES']

mean = sample.mean()
std = sample.std()
n = len(sample)

z = (mean - mean) / (std / np.sqrt(n))
p = 2 * (1 - stats.norm.cdf(abs(z)))

print("Z-score:", z)
print("P-value:", p)

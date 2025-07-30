# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Paths
data_path = r'X:\StartupFundingAnalysis\StartupFundingAnalysis\data\startup_funding.csv'
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()  # Fallback for interactive/Jupyter
output_path = os.path.join(os.path.dirname(script_dir), 'outputs')

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path, encoding='utf-8')
print("Column names in your CSV:", df.columns.tolist())

# === Data Cleaning ===
column_mapping = {
    'Date dd/mm/yyyy': 'Date',
    'Amount in USD': 'AmountInUSD',
    'Startup Name': 'StartupName',
    'Industry Vertical': 'IndustryVertical'
}

for old_name, new_name in column_mapping.items():
    if old_name in df.columns:
        df.rename(columns={old_name: new_name}, inplace=True)

df.columns = df.columns.str.strip().str.replace(' ', '')

# Convert Date
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df['Year'] = df['Date'].dt.year
else:
    raise ValueError("Missing 'Date' column.")

# Convert Amount
if 'AmountInUSD' in df.columns:
    df['AmountInUSD'] = pd.to_numeric(df['AmountInUSD'].astype(str).str.replace(',', ''), errors='coerce')
    df['AmountInUSD'] = df['AmountInUSD'].fillna(df['AmountInUSD'].median())
else:
    raise ValueError("Missing 'AmountInUSD' column.")

# Handle missing industries
df['IndustryVertical'] = df.get('IndustryVertical', pd.Series(['Unknown'] * len(df))).fillna('Unknown')

# === 1. Horizontal Bar Chart: Total Funding per Year ===
funding_per_year = df.groupby('Year')['AmountInUSD'].sum()
plt.figure(figsize=(10, 8))
colors = plt.cm.tab20(np.linspace(0, 1, len(funding_per_year)))
bars = plt.barh(funding_per_year.index, funding_per_year.values, color=colors, height=0.7, edgecolor='white', linewidth=1.5)
plt.title('Total Funding per Year', fontsize=14, fontweight='bold')
plt.xlabel('Total Funding (USD)', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.gca().set_facecolor('#f5f5f5')

for bar in bars:
    plt.text(bar.get_width() + (bar.get_width()*0.01), bar.get_y() + bar.get_height()/2,
             f'${bar.get_width():,.0f}', va='center')

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'funding_per_year.png'), dpi=300)
plt.close()

# === 2. Stacked Bar: Startups Funded per Year ===
startups_per_year = df.groupby('Year')['StartupName'].count()
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

bottom = startups_per_year * 0.4
top = startups_per_year * 0.6

bottom_bars = ax.bar(startups_per_year.index, bottom, color='#EA4335', width=0.8, edgecolor='white', linewidth=0.5)
top_bars = ax.bar(startups_per_year.index, top, bottom=bottom, color='#4285F4', width=0.8, edgecolor='white', linewidth=0.5)

for i, v in enumerate(startups_per_year.values):
    ax.text(startups_per_year.index[i], v + 0.5, f'{int(v)}', ha='center', va='bottom', fontweight='bold')

ax.set_title('Number of Startups Funded per Year', fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Startups')
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, startups_per_year.max() * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'stacked_startups_per_year.png'), dpi=300)
plt.close()

# === 3. Pie Chart: Top 5 Industries by Funding ===
industry_funding = df.groupby('IndustryVertical')['AmountInUSD'].sum().nlargest(5)
plt.figure(figsize=(10, 8))
colors = ['#ff9e4a', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
plt.pie(industry_funding, labels=None, autopct='%1.1f%%', startangle=90,
        colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
plt.title('Top 5 Industries by Funding', fontsize=14, fontweight='bold')
plt.legend(industry_funding.index, loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'industry_funding.png'), dpi=300)
plt.close()

# === 4. Line Chart: Average Funding per Startup per Year ===
avg_funding = df.groupby('Year')['AmountInUSD'].mean()
plt.figure(figsize=(10, 6))
plt.plot(avg_funding.index, avg_funding.values, marker='o', linestyle='-', color='#2E86AB', linewidth=2.5)
plt.title('Average Funding per Startup per Year', fontsize=14, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Average Funding (USD)')
plt.grid(True, linestyle='--', alpha=0.4)
plt.gca().set_facecolor('#fdfdfd')
for x, y in zip(avg_funding.index, avg_funding.values):
    plt.text(x, y + (max(avg_funding)*0.015), f"${y:,.0f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'avg_funding_per_year.png'), dpi=300)
plt.close()


# === 6. Horizontal Bar Chart: Top 10 Cities by Total Funding ===

# Clean City column if exists
if 'CityLocation' in df.columns:
    df['CityLocation'] = df['CityLocation'].astype(str).str.strip().str.title()
    df['CityLocation'] = df['CityLocation'].replace({
        'Delhi': 'New Delhi',
        'Bangalore': 'Bengaluru',
        'Mumbai': 'Mumbai',
        'Gurgaon': 'Gurugram',
        'Hyderabad': 'Hyderabad'
    })

    city_funding = df.groupby('CityLocation')['AmountInUSD'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 8))
    bars = plt.barh(city_funding.index[::-1], city_funding.values[::-1], color=plt.cm.viridis(np.linspace(0.2, 0.8, 10)))
    plt.title('Top 10 Cities by Total Funding', fontsize=14, fontweight='bold')
    plt.xlabel('Total Funding (USD)', fontsize=12)
    plt.ylabel('City', fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.gca().set_facecolor('#f5f5f5')

    for bar in bars:
        plt.text(bar.get_width() + (bar.get_width() * 0.01), bar.get_y() + bar.get_height()/2,
                 f'${bar.get_width():,.0f}', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'top_cities_by_funding.png'), dpi=300)
    plt.close()

else:
    print("City column ('CityLocation') not found in the dataset.")

import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ??? IoT IDS - Day 1: EDA\n",
                "## 1. Load the Stratified Sample Data\n",
                "We loaded a 10% stratified sample from 10 files to keep memory low (~2.3M rows)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "plt.style.use('dark_background')\n",
                "\n",
                "# Load the sample we created earlier\n",
                "df = pd.read_parquet('processed/eda_sample.parquet')\n",
                "print(f'Dataset Shape: {df.shape}')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Check Missing Values & Data Types"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "missing_values = df.isnull().sum()\n",
                "print('Missing Values:\\n', missing_values[missing_values > 0])\n",
                "\n",
                "# Check for Infinite values\n",
                "inf_counts = np.isinf(df.select_dtypes(include=np.number)).sum()\n",
                "print('\\nInfinite Values:\\n', inf_counts[inf_counts > 0])\n",
                "\n",
                "print('\\nData Types:\\n', df.dtypes.value_counts())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Analyze the 34-Class Distribution (Extreme Imbalance)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(12, 8))\n",
                "class_counts = df['label'].value_counts()\n",
                "sns.barplot(y=class_counts.index, x=class_counts.values, palette='viridis')\n",
                "plt.title('Distribution of 34 Attack Classes (Log Scale)')\n",
                "plt.xlabel('Count')\n",
                "plt.ylabel('Attack Type')\n",
                "plt.xscale('log')  # Log scale because of extreme imbalance\n",
                "plt.tight_layout()\n",
                "plt.savefig('figures/34_class_distribution.png', dpi=300)\n",
                "plt.show()\n",
                "\n",
                "print(class_counts)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Map to 8-Class Family Taxonomy and Visualize"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def map_to_family(label):\n",
                "    if label == 'BenignTraffic': return 'Benign'\n",
                "    elif label.startswith('DDoS'): return 'DDoS'\n",
                "    elif label.startswith('DoS'): return 'DoS'\n",
                "    elif label.startswith('Mirai'): return 'Mirai'\n",
                "    elif label.startswith('Recon'): return 'Recon'\n",
                "    elif label in ['DNS_Spoofing', 'MITM-ArpSpoofing']: return 'Spoofing'\n",
                "    elif label in ['BrowserHijacking', 'CommandInjection', 'SqlInjection', 'Uploading_Attack', 'XSS']: return 'Web'\n",
                "    elif label == 'DictionaryBruteForce': return 'BruteForce'\n",
                "    return 'Unknown'\n",
                "\n",
                "df['label_family'] = df['label'].apply(map_to_family)\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "family_counts = df['label_family'].value_counts()\n",
                "sns.barplot(x=family_counts.index, y=family_counts.values, palette='rocket')\n",
                "plt.title('Distribution of 8 Attack Families (Log Scale)')\n",
                "plt.ylabel('Count')\n",
                "plt.yscale('log')\n",
                "plt.xticks(rotation=45)\n",
                "plt.tight_layout()\n",
                "plt.savefig('figures/8_class_distribution.png', dpi=300)\n",
                "plt.show()\n",
                "\n",
                "print(family_counts)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Analyze Feature Correlations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(14, 12))\n",
                "# Compute correlation matrix for numeric features\n",
                "numeric_df = df.select_dtypes(include=np.number)\n",
                "corr = numeric_df.corr()\n",
                "\n",
                "# Show only correlations > 0.8 to identify redundant features\n",
                "high_corr = corr[(corr >= 0.8) | (corr <= -0.8)]\n",
                "sns.heatmap(high_corr, cmap='coolwarm', center=0, cbar_kws={'shrink': 0.8})\n",
                "plt.title('High Feature Correlations (>|0.8|)')\n",
                "plt.tight_layout()\n",
                "plt.savefig('figures/high_correlation_heatmap.png', dpi=300)\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs('figures', exist_ok=True)
with open('01_EDA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

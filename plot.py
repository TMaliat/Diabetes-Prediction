import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Loop through each column and save its plot
for column in diabetes_dataset.columns:
    plt.figure(figsize=(8, 6))  # Set figure size
    sns.histplot(diabetes_dataset[column], kde=True, bins=30, color='blue')  # Plot histogram with density
    plt.title(f'Distribution of {column}')  # Add a title
    plt.xlabel(column)  # Label x-axis
    plt.ylabel('Frequency')  # Label y-axis
    
    # Save the plot as an image
    plt.savefig(f'{column}_plot.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to save memory

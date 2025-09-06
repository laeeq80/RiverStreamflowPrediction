import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to plot regressions
def plot_temperature_rainfall_regressions(df, order=4):
    """
    Create regression plots for Minimum Temperature, Average Temperature, 
    Maximum Temperature, and Rainfall against Water Discharge.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing the columns 'Minimum Temperature', 
                          'Average Temperature', 'Maximum Temperature', 'Rainfall', 
                          and 'water Discharge'
    order (int): Order of polynomial regression (default=4)
    """
    plot_configs = [
        {'x': 'Minimun_Temperature', 'color': 'green', 'title': 'Min Temperature vs Water Discharge'},
        {'x': 'Average_Temperature', 'color': 'purple', 'title': 'Avg Temperature vs Water Discharge'},
        {'x': 'Maximum_Temperature', 'color': 'blue', 'title': 'Max Temperature vs Water Discharge'},
        {'x': 'Rainfall', 'color': 'red', 'title': 'Rainfall vs Water Discharge'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    
    axes = axes.ravel()

    for idx, config in enumerate(plot_configs):
        sns.regplot(
            x=df[config['x']],
            y=df['water_Discharge'],
            order=order,
            scatter_kws={'color': config['color']},
            ax=axes[idx]
        )
        axes[idx].set_title(config['title'])
    
    plt.tight_layout()
    plt.show()

# Load cleaned data
try:
    df = pd.read_csv('data/intermidiate/cleaned.csv')  # Relative path from cleaned_data_set/
    print("Cleaned data loaded successfully.")
except FileNotFoundError:
    print("Error: 'data/intermidiate/cleaned.csv' not found. Please run data.py first to generate the file.")
    df = None
except Exception as e:
    print(f"Error loading data: {e}")
    df = None

# Execute plotting if data is loaded successfully
if df is not None:
    plot_temperature_rainfall_regressions(df, order=4)
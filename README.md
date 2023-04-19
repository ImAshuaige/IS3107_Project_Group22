# IS3107_Project

### Objective
#### Dag.py

#### EDA.py
The objective of this Python file is to provide exploratory data analysis and data visualization for the obtained datasets. These techniques will help the user gain an overall and better understanding of the relationships, distributions, quality, and patterns of the data. For example, this Python file will analyze the general trend of Bitcoin price over time, as well as the correlation between tweet sentiment scores and the Bitcoin price etc. The insights gained from the EDA will then be used for prediction models later.

#### Dashboard.py

### Tools & Technologies
- Storage - [PostgreSQL](https://www.postgresql.org/)
- Orchestration - [Airflow](https://airflow.apache.org/)
- Language - [Python](https://www.python.org/)

### Data 
- bitcoin.csv - Provides data on Bitcoin quotes
- Finalised_Sentiments.csv - Provide data on Bitcoin-related Tweets' sentiments
### Setup

### Instructions on Running the Files
1. Download all the files in the GitHub repository. 
2. The order of running the files should be: 1)Dag.py -> 2)EDA.py -> 3)Dashboard.py

### Detailed Steps for Running Each File
#### 1)Dag.py

#### 2)EDA.py
1. Open your web browser and go to https://colab.research.google.com/.
2. Convert the EDA.**py** file to a Jupyter Notebook format by renaming it to EDA.**ipynb**.
3. Click on "Upload" in the "Upload or Create Notebook" dialog to upload the Python file (EDA.ipynb).
4. Once the notebook is open in Google Colab, click on the folder icon in the left sidebar to open the "Files" tab.
5. Click the "Upload" button (the icon with an upward arrow) in the "Files" tab to upload the two CSV files (**bitcoin.csv** and **Finalised_Sentiments.csv**).
6. Replace 
   ```pd.read_csv('/content/drive/MyDrive/IS3107/bitcoin.csv')``` and ```pd.read_csv('/content/drive/MyDrive/IS3107/Finalised_Sentiments.csv')```
   with the updated file paths, such as: ```pd.read_csv('/content/bitcoin.csv')``` and ```pd.read_csv('/content/Finalised_Sentiments.csv')```
7. Run the entire notebook by clicking on "Runtime" in the top menu, and then selecting "Run all". Alternatively, you can run each cell one by one by clicking the "Play" button on the left side of each cell.
8. Once the script finishes running, you'll see the plots.

#### 3)Dashboard.py

![Bitcoin Predictions Dashboard](https://i.ibb.co/kBPVnpX/bitcoin-dashboard.png)


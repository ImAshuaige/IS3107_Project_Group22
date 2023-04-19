# IS3107_Project

### Objective
#### Dag.py
#### Dashboard.py
#### EDA.py

### Tools & Technologies
- Storage - [PostgreSQL](https://www.postgresql.org/)
- Orchestration - [Airflow](https://airflow.apache.org/)
- Language - [Python](https://www.python.org/)

### Data 

### Setup

### Instructions on Running the Files
1. Download all the files in the GitHub repository. 
### Detailed Steps for Each File
#### Dag.py
#### Dashboard.py
#### EDA.py
1. Open your web browser and go to https://colab.research.google.com/.
2. Convert the EDA.py file to a Jupyter Notebook format by renaming it to EDA.ipynb.
3. Click on "Upload" in the "Upload or Create Notebook" dialog to upload the Python file (EDA.ipynb).
4. Once the notebook is open in Google Colab, click on the folder icon in the left sidebar to open the "Files" tab.
5. Click the "Upload" button (the icon with an upward arrow) in the "Files" tab to upload the two CSV files (bitcoin.csv and Finalised_Sentiments.csv).
6. Replace 
   ```pd.read_csv('/content/drive/MyDrive/IS3107/bitcoin.csv')```
   ```pd.read_csv('/content/drive/MyDrive/IS3107/Finalised_Sentiments.csv')```
   with the updated file paths, such as:
   pd.read_csv('/content/bitcoin.csv')
   pd.read_csv('/content/Finalised_Sentiments.csv')
7. Run the entire notebook by clicking on "Runtime" in the top menu, and then selecting "Run all". Alternatively, you can run each cell one by one by clicking the "Play" button on the left side of each cell.
8. Once the script finishes running, you'll see the plots.

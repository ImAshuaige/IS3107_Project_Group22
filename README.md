# IS3107_Project

### Objective
#### Dag.py
The objective of this Python file is to collect, transform, and load data related to Bitcoin prices and tweets into a PostgreSQL database. This data is loaded from 3 distinct sources including Kaggle, CoinMarketCap API and the Snscrape command line tool. Additionally, this file will perform sentiment analysis on the collected tweets using the VADER sentiment analyzer. The collected data will be used for exploratory data analysis and data visualization to gain insights into the general trends and correlations between certain variables.

#### EDA.py
The objective of this Python file is to provide exploratory data analysis and data visualization for the obtained datasets. These techniques will help the user gain an overall and better understanding of the relationships, distributions, quality, and patterns of the data. For example, this Python file will analyze the general trend of Bitcoin price over time, as well as the correlation between tweet sentiment scores and the Bitcoin price etc. The insights gained from the EDA will then be used for prediction models later.

#### Dashboard.py
This Python file is used to run the dashboard on localhost:8051. On the dashboard, the predictions from our Machine Learning models can be seen in graph form. Additionally, the r2 score for each model is also displayed on the dashboard for each of the models used. This dashboard utilises the PostgreSQL database tables that would have been updated and filled after running the Dag.py file.

### Tools & Technologies
- Storage - [PostgreSQL](https://www.postgresql.org/)
- Orchestration - [Airflow](https://airflow.apache.org/)
- Language - [Python](https://www.python.org/)

### Data 
- bitcoin.csv - Provides data on Bitcoin quotes
- initial_tweets.json - Provides the initial Tweet data (from 13/03/23 to 12/04/23) we used before starting the streaming process
- Finalised_Sentiments.csv - Provide data on Bitcoin-related Tweets' sentiments
### Setup

### Instructions on Running the Files
1. Download all the files in the GitHub repository. 
2. The order of running the files should be: 1)Dag.py -> 2)EDA.py -> 3)Dashboard.py
3. Create a Python environment by running ```python3 -m venv env``` and activate the environment using ```source env/bin/activate```
4. In the environment created, install the required packages by running ```pip install -r requirements.txt```

### Detailed Steps for Running Each File
#### 1)Dag.py
1. Download [PGAdmin 4](https://www.pgadmin.org/download/) to easily access the tables that will be created after running this file.
2. Set up a PostgreSQL connection using the following structure: ```postgresql+psycopg2://postgres:password@localhost:5432/IS3107_Project``` such that the database has these specifications [**password**: password, **port_number**: 5432, **database_name**: IS3107_Project] 
3. If your connection is different from the connection defined in step 3, it needs to be updated within the following tasks: ```transform_data(), task_data_upload(table_name, data), create_bitcoin_tables(), sentiment_analysis(), bitcoin_stream(), tweets_stream() and sentiment_task_data_upload(table_name, data)```.
4. Save the updated code as a python file on your local machine within the ```dags``` folder in your airflow directory
5. Unzip the ```initial_tweets.json``` file on your local machine and update the directory to ```initial_tweets.json``` in the ```transform_data()``` function
6. Within the python environment activated earlier, navigate to the dags folder where the ```IS3107_project.py``` file was saved 
7. Run the following command to execute the file and trigger the DAG: ```airflow dags test IS3107_project```
8. Once this command is successfully completed, 3 tables will appear within PostgreSQL & PGAdmin 4: ```bitcoin_tweet, bitcoin_prices and bitcoin_tweets_sentiment```

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

#### 3)dashboard.py
1. Replace the variables "password" and "database_schema" with the PostgreSQL password and schema name on your local computer.
2. Run the dashboard.py script within the Python environment created earlier using ```python3 dashboard.py```
3. Go to (http://127.0.0.1:8051) to view the dashboard below.

![Bitcoin Predictions Dashboard](https://i.ibb.co/r4CWjGL/bitcoin-dashboard.png)


# Disaster Response Pipeline Project

This project is for the Data Engineering section for Udacity Data Science Nanodegree. The bassis of this project is to look at data provided by FigureEight. The data are tweets and texts that were sent during real world disasters and can be labeled into at least one of 36 categories. 

This project required 3 steps:
  1. Create ETL process of data from CSV files and upload cleansed data to a SQLite database.
  2. create machine learning pipeline to analyze messages and optomize model to correctly classify labels for that text.
  3. Create web aplication that can show 2 graphs of overviews of the messages, as well as a input bar that could read a message and correctly classify what label it would belong to.

### File Descriptions:

**app**

  - template
    - master.html `main page of web app`
    - go.html `classification result page of web app`
  - run.py # `Flask file that runs app`

**data**

  - disaster_categories.csv `data to process`
  - disaster_messages.csv  `data to process`
  - process_data.py  `data cleaning pipeline`
  - DisasterResponse.db  `database saved after cleansed`

**models**

  - train_classifier.py  ` machine learning pipeline`
  - classifier.pkl  `saved model`


### Instructions:
The way you can run this project:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. To find your environment of your workspace  

    -  open new terminal and print: 
`env|grep WORK `

#### This will print: 
`WORKSPACEDOMAIN=udacity-student-workspaces.com WORKSPACEID=view6914b2f4`

    - identify your website link by doing substitutions: https://WORKSPACEID-3001.WORKSPACEDOMAIN 
    your website link will be: https://view6914b2f4-3001.udacity-student-workspace
    
#### The App will look like:
![image](https://user-images.githubusercontent.com/70536290/147861198-63764326-9115-49e0-b06f-8f0eb07a10bc.png)

    



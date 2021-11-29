# X0PA Task
## TASK 1
*Your task is to write the following queries in SQL based on the Database Schema (see [sqlSchema.sql](https://github.com/koulakhilesh/x0pa-task/blob/master/sqlSchema.sql)) . The database schema is written in PostgreSQL. However, you are free to use any SQL dialect but please specify the  SQL dialect you are using*.
**USING MYSQL**

a) *Write a query to print the total number of open positions (open positions = vacancies - # of hired candidates) for each company as of the current date.*

**See [TASK 1a](https://github.com/koulakhilesh/x0pa-task/blob/master/TASK1_A.sql) for solution.**

b) *Write a query to print the industry, company name, and total number of local applications (Singapore candidates that applied to a job in the company) for companies with the highest number of different jobs in each industry. For an example, ABC company has 80 different jobs (the highest in the agriculture industry) and has 1,000 Singapore candidates applying to its jobs. The query would print {‘industry: ‘Agriculture’, ‘company’:‘ABC’, ‘local_applications’:1000} for this record.*

**See [TASK 1b](https://github.com/koulakhilesh/x0pa-task/blob/master/TASK1_B.sql) for solution.**

## TASK 2 
### *Problem Statement*
*The core of X0PA products often involve APIs around machine learning algorithms and NLP. In real world data especially in human resources, they come in different schemas and it is important to standardize them into one. For this problem, we will be looking into classifying any job titles into job functions. Initially, we tried to train a model that classify job titles into all job functions. However, we found out that the information technology job functions are too general and it is important for us to break them further down into various subclasses. By breaking them down, we are then able to match candidates to jobs to a higher accuracy. This problem will test your ability to build a basic NLP model based on a given dataset. Your end task will be to:* 

* *develop a model to predict one of the 16 classes (see Variables Schema);*

**Main File with class object is written in python and available at [TASK_2.py](https://github.com/koulakhilesh/x0pa-task/blob/master/Task2.py). It consist of  functions for initialization, data acquisition, cleaning of the text, and batch training, validation evaluation, batch testing, single prediction. See [Task2-docstring](https://github.com/koulakhilesh/x0pa-task/blob/master/Task2_docstring.txt). GradientBoostingClassifier is used for modeling classifier for this task**

* *provide justification for model evaluation and report your results;*

**Since it multi-classification, confusion matrix and F1 score is obtained on validation set. See [Task2-Notebook](https://github.com/koulakhilesh/x0pa-task/blob/master/Task2_NB.ipynb) for display for evaluation results of validation set. Also results for the testing data-set can be obtained from [Test_Results](https://github.com/koulakhilesh/x0pa-task/blob/master/test_y_pred.csv)**

* *deploy your model in the form of an API endpoint (any API framework will do, but FastAPI is
preferred).*
**Ah! for this I am unable to implement FastAPI as I haven't used it till now in my work. But this can be implemented with functions already developed in [TASK_2.py](https://github.com/koulakhilesh/x0pa-task/blob/master/Task2.py) (also see [Task2-Notebook](https://github.com/koulakhilesh/x0pa-task/blob/master/Task2_NB.ipynb)) . Had lot of trouble with Docker, Uvicorn and FastAPI, apologies!**

**Finally please see the [requirements.txt](https://github.com/koulakhilesh/x0pa-task/blob/master/requirements.txt) for all the required python library needed to run code for task 2**

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: akhilesh.koul
"""

import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

class whatjob():
    
    def __init__(self,train_path='data/x0pa_ds_interview_round_2_train.xlsx',test_path='data/x0pa_ds_interview_round_2_test.xlsx',inputFeat='Job Title',Target='Type'):
        """
        Initialization for whatJob 

        Parameters
        ----------
        train_path : STR, optional
            Path for training set. The default is 'data/x0pa_ds_interview_round_2_train.xlsx'.
        test_path : STR, optional
            Path for testing set. The default is 'data/x0pa_ds_interview_round_2_test.xlsx'.
        inputFeat : STR, optional
            Input Feature Column Name. The default is 'Job Title'.
        Target : STR, optional
            Target Column Name. The default is 'Type'.

        Returns
        -------
        None.

        """

        self.pkl_path='pkl_files/' 
        self.df_train_path=train_path
        self.df_test_path=test_path
        self.train_df=pd.read_excel(self.df_train_path)
        self.test_df=pd.read_excel(self.df_test_path)
        self.inputFeat=inputFeat
        self.Target=Target
    
    def getData(self):
        """
        Helper Function to get training, validation and testing data

        Returns
        -------
        train_set : PANDAS DATAFRAME
            Training data-set.
        validation_set : PANDAS DATAFRAME
            Validation data-set.
        PANDAS DATAFRAME
            Testing data-set.

        """
        
        train_set, validation_set=train_test_split(self.train_df,test_size=0.2, random_state=42, shuffle=True)
        return train_set,validation_set, self.test_df   
    
    def cleanText(self,df,train_flag=True):
        """
        Helper function to clean the text based on commonsense and nltk library

        Parameters
        ----------
        df : PANDAS DATAFRAME
            Pandas DataFrame to be cleaned.
        train_flag : BOOL, optional
            True if cleaning text for training set, else False. The default is True.

        Returns
        -------
        df_new : PANDAS DATAFRAME
            Cleaned Pandas DataFrame.

        """
        
        df_new=df.copy() #copy to avoid SettingwithCopyWarning
        #remove non_ascii
        df_new[self.inputFeat] = df_new[self.inputFeat].apply(lambda text: self.remove_non_ascii(text))
        #remove number
        df_new[self.inputFeat] = df_new[self.inputFeat].apply(lambda text: self.remove_num(text))
        #lower case
        df_new[self.inputFeat] = df_new[self.inputFeat].str.lower()
        #remove_punctuation
        df_new[self.inputFeat] = df_new[self.inputFeat].apply(lambda text: self.remove_punctuation(text,train=train_flag))
        #remove_stopwords
        df_new[self.inputFeat] = df_new[self.inputFeat].apply(lambda text: self.remove_stopwords(text,train=train_flag)) 
        #lemmatize_words
        df_new[self.inputFeat] = df_new[self.inputFeat].apply(lambda text: self.lemmatize_words(text))
        #remove_extra_whitespace_tabs
        df_new[self.inputFeat] = df_new[self.inputFeat].apply(lambda text: self.remove_extra_whitespace_tabs(text))
        
        return df_new

    def remove_non_ascii(self,text):
        """
        Helper Function to remove non ascii characters
        
        
        Parameters
        ----------
        text : STR
            Original Text.

        Returns
        -------
        STR
            Cleaned Text.

        """
        return "".join(c for c in text if ord(c)<128)
    
    def remove_num(self,text):
        """
        Helper Function to remove numbers
        
        Parameters
        ----------
        text : STR
            Original Text.

        Returns
        -------
        STR
            Cleaned Text.

        """
        return re.sub(r'\d+', '', text)
    
    def remove_punctuation(self,text,train=False):
        """
        Helper Function to remove punctuation

        Parameters
        ----------
        text : STR
            Original Text.
        train : BOOL, optional
              True, if cleaning text for training set, else False. The default is True.

    
        Returns
        -------
        STR
            Cleaned Text.


        """
        
        if train==True:
            PUNCT_TO_REMOVE = '!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~”“`'
            with open(self.pkl_path+'PUNCT_TO_REMOVE.pkl', 'wb') as f:
                pickle.dump(PUNCT_TO_REMOVE, f)
                
        with open(self.pkl_path+'PUNCT_TO_REMOVE.pkl', 'rb') as f:
                PUNCT_TO_REMOVE = pickle.load(f)   
               
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


    def remove_stopwords(self,text,train=False):
        """
        Helper Function to remove stop words

        Parameters
        ----------
        text : STR
            Original Text.
        train : BOOL, optional
              True, if cleaning text for training set, else False. The default is True.

    
        Returns
        -------
        STR
            Cleaned Text.


        """
       
        STOPWORDS = set(stopwords.words('english'))
      
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    
    
    def lemmatize_words(self,text):
        """
        Helper Function to lemmatize words
        
        Parameters
        ----------
        text : STR
            Original Text.

        Returns
        -------
        STR
            Cleaned Text.
            
        """
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
        pos_tagged_text = nltk.pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
   
    def remove_extra_whitespace_tabs(self,text):
        """
        Helper Function to remove extra white-space tabs
        
        Parameters
        ----------
        text : STR
            Original Text.

        Returns
        -------
        STR
            Cleaned Text.
            
        """
        pattern = r'^\s*|\s\s*'
        return re.sub(pattern, ' ', text).strip()
    
    
    def batchTraining(self,traindf):
        """
        Function to do batch training based on the training data-set

        Parameters
        ----------
        traindf : PANDAS DATAFRAME
            Training DataFrame.

        Returns
        -------
        clf : OBJECT
             Fitted Classifier.

        """
        
        tfidfvectorizer = TfidfVectorizer()
        train_text_vector = tfidfvectorizer.fit_transform(traindf[self.inputFeat])
     
        with open(self.pkl_path+'tfidfvectorizer.pkl', 'wb') as f:
               pickle.dump(tfidfvectorizer, f)

        tfidftransformer = TfidfTransformer()
        train_text_tfidf = tfidftransformer.fit_transform(train_text_vector)
       
        with open(self.pkl_path+'tfidftransformer.pkl', 'wb') as f:
               pickle.dump(tfidftransformer, f)
       
        X_train=train_text_tfidf
        y_train=np.array(traindf[self.Target])
        clf = GradientBoostingClassifier(n_estimators=200, verbose=1,validation_fraction=0.2,n_iter_no_change=20).fit(X_train, y_train)   
       
             
        with open(self.pkl_path+'clf.pkl', 'wb') as f:
               pickle.dump(clf, f)
        return clf   
    
    def valBatchEval(self,valdf):
        """
        Function to do evaluate trained model on validation set and obtain/plot confusion matrix report.

        Parameters
        ----------
        valdf : PANDAS DATAFRAME
            Validation DataFrame.

        Returns
        -------
        y_true : NUMPY ARRAY
            Numpy array of True values of the target column.
        y_peed : NUMPY ARRAY
            Numpy array of Predicted values of the target column.

        """
               
        with open(self.pkl_path+'tfidfvectorizer.pkl', 'rb') as f:
            tfidfvectorizer = pickle.load(f)  
       
        with open(self.pkl_path+'tfidftransformer.pkl', 'rb') as f:
            tfidftransformer = pickle.load(f)  
        
                
        df_text_vector = tfidfvectorizer.transform(valdf[self.inputFeat])
        df_text_tfidf = tfidftransformer.transform(df_text_vector)
        
        with open(self.pkl_path+'clf.pkl', 'rb') as f:
            clf = pickle.load(f)  
    
        y_peed = clf.predict(df_text_tfidf)
        y_true=np.array(valdf[self.Target])
        print(metrics.classification_report(y_true, y_peed, digits=3))
        y_unique = np.unique(y_peed)
        cm = metrics.confusion_matrix(y_true, y_peed)
        cm_df = pd.DataFrame(cm,
                             index = y_unique, 
                             columns = y_unique)

        #Plotting the confusion matrix
        plt.figure(figsize=(15,15))
        sns.heatmap(cm_df, annot=True,fmt='g')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.show()
        
        return y_true,y_peed
    
    def testBatch(self,testdf):
        """
         Function to do batch testing based on the trained model


        Parameters
        ----------
        testdf : PANDAS DATAFRAME
            Testing DataFrame.


        Returns
        -------
        y_peed : NUMPY ARRAY
            Numpy array of Predicted values for the testing data-set.

        """
               
        with open(self.pkl_path+'tfidfvectorizer.pkl', 'rb') as f:
            tfidfvectorizer = pickle.load(f)  
       
        with open(self.pkl_path+'tfidftransformer.pkl', 'rb') as f:
            tfidftransformer = pickle.load(f)  
        
                
        df_text_vector = tfidfvectorizer.transform(testdf[self.inputFeat])
        df_text_tfidf = tfidftransformer.transform(df_text_vector)
        
        with open(self.pkl_path+'clf.pkl', 'rb') as f:
            clf = pickle.load(f)  
    
        y_peed = clf.predict(df_text_tfidf)
        
    
        return y_peed  
    
    def getPrediction(self,text):
        """
        Function to run classifier on single user input

        Parameters
        ----------
        text : STR
            User input for the Job Title.

        Returns
        -------
        y_peed : STR
            Predicted Job Type based on the trained classifier.

        """
   
        text=self.remove_non_ascii(text)
        text=self.remove_num(text)
        text=text.lower()
        text=self.remove_punctuation(text)
        text=self.remove_stopwords(text)
        text=self.lemmatize_words(text)
        text=self.remove_extra_whitespace_tabs(text)
        text_list=[]
        text_list.append(text)
        df_single=pd.DataFrame()
        df_single[self.inputFeat]=text_list
       
        with open(self.pkl_path+'tfidfvectorizer.pkl', 'rb') as f:
            tfidfvectorizer = pickle.load(f)  
       
        with open(self.pkl_path+'tfidftransformer.pkl', 'rb') as f:
            tfidftransformer = pickle.load(f)  
        
                
        df_text_vector = tfidfvectorizer.transform(df_single[self.inputFeat])
        df_text_tfidf = tfidftransformer.transform(df_text_vector)
        
        with open(self.pkl_path+'clf.pkl', 'rb') as f:
            clf = pickle.load(f)  
    
        y_pred = clf.predict(df_text_tfidf)
  
        return y_pred
    
    
if __name__ == '__main__':
    #initialization
    whatJobClass=whatjob()
    
    #getData
    train_df,valid_df,test_df=whatJobClass.getData()   
   
    #batchTraining
    # train_df_clean=whatJobClass.cleanText(train_df,train_flag=True)
    # clf=whatJobClass.batchTraining(train_df_clean)
    
    
    #valBatchEval
    # valid_df_clean=whatJobClass.cleanText(valid_df,train_flag=False)
    # _,_=whatJobClass.valBatchEval(valid_df_clean)
    
    #testBatch
    test_df_clean=whatJobClass.cleanText(test_df,train_flag=False)
    y_pred=whatJobClass.testBatch(test_df_clean)
    pd.DataFrame(y_pred).to_csv("test_y_pred.csv")

    #getPrediction
    # job_title='Technical Lead (QA Automation)'
    # y_pred_single=whatJobClass.getPrediction(job_title)
    # print(y_pred_single)
    


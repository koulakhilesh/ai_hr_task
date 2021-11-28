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
        

        Parameters
        ----------
        train_path : TYPE, optional
            DESCRIPTION. The default is 'data/x0pa_ds_interview_round_2_train.xlsx'.
        test_path : TYPE, optional
            DESCRIPTION. The default is 'data/x0pa_ds_interview_round_2_test.xlsx'.
        inputFeat : TYPE, optional
            DESCRIPTION. The default is 'Job Title'.
        Target : TYPE, optional
            DESCRIPTION. The default is 'Type'.

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
        

        Returns
        -------
        train_set : TYPE
            DESCRIPTION.
        validation_set : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        
        train_set, validation_set=train_test_split(self.train_df,test_size=0.2, random_state=42, shuffle=True)
        return train_set,validation_set, self.test_df   
    
    def cleanText(self,df,train_flag=True):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        train_flag : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        df_new : TYPE
            DESCRIPTION.

        """
        
        df_new=df.copy() #copy to avoid SettingwithCopyWarning
        #remove non_ascii
        df_new[self.inputFeat] = df_new[self.inputFeat].apply(lambda text: self.remove_non_ascii(text))
        #remove number
        df_new[self.inputFeat] = df_new[self.inputFeat].apply(lambda text: self.remove_num(text))
        #lowwer case
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
        

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return "".join(c for c in text if ord(c)<128)
    
    def remove_num(self,text):
        """
        

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return re.sub(r'\d+', '', text)
    
    def remove_punctuation(self,text,train=False):
        """
        

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.
        train : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

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
        

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.
        train : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        STOPWORDS = set(stopwords.words('english'))
      
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    
    
    def lemmatize_words(self,text):
        """
        

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
        pos_tagged_text = nltk.pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
   
    def remove_extra_whitespace_tabs(self,text):
        """
        

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        pattern = r'^\s*|\s\s*'
        return re.sub(pattern, ' ', text).strip()
    
    
    def batchTraining(self,traindf):
        """
        

        Parameters
        ----------
        traindf : TYPE
            DESCRIPTION.

        Returns
        -------
        clf : TYPE
            DESCRIPTION.

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
        

        Parameters
        ----------
        valdf : TYPE
            DESCRIPTION.

        Returns
        -------
        y_true : TYPE
            DESCRIPTION.
        y_pred : TYPE
            DESCRIPTION.

        """
               
        with open(self.pkl_path+'tfidfvectorizer.pkl', 'rb') as f:
            tfidfvectorizer = pickle.load(f)  
       
        with open(self.pkl_path+'tfidftransformer.pkl', 'rb') as f:
            tfidftransformer = pickle.load(f)  
        
                
        df_text_vector = tfidfvectorizer.transform(valdf[self.inputFeat])
        df_text_tfidf = tfidftransformer.transform(df_text_vector)
        
        with open(self.pkl_path+'clf.pkl', 'rb') as f:
            clf = pickle.load(f)  
    
        y_pred = clf.predict(df_text_tfidf)
        y_true=np.array(valdf[self.Target])
        print(metrics.classification_report(y_true, y_pred, digits=3))
        y_unique = np.unique(y_pred)
        cm = metrics.confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm,
                             index = y_unique, 
                             columns = y_unique)

        #Plotting the confusion matrix
        plt.figure(figsize=(15,15))
        sns.heatmap(cm_df, annot=True,fmt='g')
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()
        
        return y_true,y_pred
    
    def testBatch(self,testdf):
        """
        

        Parameters
        ----------
        testdf : TYPE
            DESCRIPTION.

        Returns
        -------
        y_pred : TYPE
            DESCRIPTION.

        """
               
        with open(self.pkl_path+'tfidfvectorizer.pkl', 'rb') as f:
            tfidfvectorizer = pickle.load(f)  
       
        with open(self.pkl_path+'tfidftransformer.pkl', 'rb') as f:
            tfidftransformer = pickle.load(f)  
        
                
        df_text_vector = tfidfvectorizer.transform(testdf[self.inputFeat])
        df_text_tfidf = tfidftransformer.transform(df_text_vector)
        
        with open(self.pkl_path+'clf.pkl', 'rb') as f:
            clf = pickle.load(f)  
    
        y_pred = clf.predict(df_text_tfidf)
        
    
        return y_pred  
    
    def testSample(self,text):
        """
        

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.

        Returns
        -------
        y_pred : TYPE
            DESCRIPTION.

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
    
    whatJobClass=whatjob()
    train_df,valid_df,test_df=whatJobClass.getData()
    
    # #training
    train_df_clean=whatJobClass.cleanText(train_df,train_flag=True)
    clf=whatJobClass.batchTraining(train_df_clean)
        
    # # #validatoin
    valid_df_clean=whatJobClass.cleanText(valid_df,train_flag=False)
    _,_=whatJobClass.valBatchEval(valid_df_clean)
  
    # # testing
    test_df_clean=whatJobClass.cleanText(test_df,train_flag=False)
    y_pred=whatJobClass.testBatch(valid_df_clean)
   
    #  single testing
    job_title='Technical Lead (QA Automation)'
    y_pred_single=whatJobClass.testSample(job_title)
    print(y_pred_single)
   
    
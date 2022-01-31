from datetime import datetime
from statistics import median

import pandas as pd
import recordlinkage
from recordlinkage.preprocessing import clean

from app import app, database as db
from app.data_transform.helpers import create_serial_sequence
from app.data_service.models import DataLoader, Table
from app.history.models import History

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron

from sklearn import tree
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import ElasticNet, LassoLars, Ridge, LinearRegression, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from flask import Markup



def _ci(*args: str):
    if len(args) == 1:
        return '"{}"'.format(str(args[0]).replace('"', '""'))
    return ['"{}"'.format(str(arg).replace('"', '""')) for arg in args]


def _cv(*args: str):
    if len(args) == 1:
        return "'{}'".format(str(args[0]).replace("'", "''"))
    return ["'{}'".format(str(arg).replace("'", "''")) for arg in args]


history = History()

 

class Regression:
    def __init__(self):
        pass

    def linear(self, schema_id, table_name, column_name,test_ratio):
        connection = db.engine.connect()
        transaction = connection.begin()
        try:
            #app.logger.error(test_ratio+"[ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR]")
            
            schema_name = 'schema-' + str(schema_id)
            df = pd.read_sql_query('SELECT * FROM {}.{};'.format(*_ci(schema_name, table_name)), db.engine)
            app.logger.error(df.head())
            df[column_name ] = df[column_name].astype('category')
            df[column_name+"converted_cat"] = df[column_name].cat.codes

            y=df[column_name+"converted_cat"].tolist()


            df=df.drop([column_name,column_name+"converted_cat"], axis=1)
            X=df.to_numpy()

            


            #Normalizing numerical features so that each feature has mean 0 and variance 1
            feature_scaler = StandardScaler()
            X_scaled = feature_scaler.fit_transform(X)



            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=float(test_ratio), random_state=0)
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            pred_test_lr= lr.predict(X_test)
            

            re="Mean Squared Error == "+ str(round(mean_squared_error(y_test,pred_test_lr),4))
            re=re+ " Root MSE == "+ str(round(np.sqrt((mean_squared_error(y_test,pred_test_lr))),4))
            re=re+ " R2-score == "+ str(round(r2_score(y_test, pred_test_lr),4))
            
            return "Results: "+str(re) 
        except Exception as e:
            transaction.rollback()
            app.logger.error("[ERROR] Couldn't apply Linear Regression on this data")
            app.logger.exception(e)
            raise e

    def elastic(self, schema_id, table_name, column_name,test_ratio):
        connection = db.engine.connect()
        transaction = connection.begin()
        try:
            #app.logger.error(test_ratio+"[ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR]")
            
            schema_name = 'schema-' + str(schema_id)
            df = pd.read_sql_query('SELECT * FROM {}.{};'.format(*_ci(schema_name, table_name)), db.engine)
            app.logger.error(df.head())
            df[column_name ] = df[column_name].astype('category')
            df[column_name+"converted_cat"] = df[column_name].cat.codes

            y=df[column_name+"converted_cat"].tolist()


            df=df.drop([column_name,column_name+"converted_cat"], axis=1)
            X=df.to_numpy()

            


            #Normalizing numerical features so that each feature has mean 0 and variance 1
            feature_scaler = StandardScaler()
            X_scaled = feature_scaler.fit_transform(X)



            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=float(test_ratio), random_state=0)
            lr = ElasticNet(alpha = 0.05)
            lr.fit(X_train, y_train)
            pred_test_lr= lr.predict(X_test)
            

            re="Mean Squared Error == "+ str(round(mean_squared_error(y_test,pred_test_lr),4))
            re=re+ " Root MSE == "+ str(round(np.sqrt((mean_squared_error(y_test,pred_test_lr))),4))
            re=re+ " R2-score == "+ str(round(r2_score(y_test, pred_test_lr),4))
            
            return "Results: "+str(re) 
        except Exception as e:
            transaction.rollback()
            app.logger.error("[ERROR] Couldn't apply ElasticNet Regression on this data")
            app.logger.exception(e)
            raise e

    def lasso(self, schema_id, table_name, column_name,test_ratio):
        connection = db.engine.connect()
        transaction = connection.begin()
        try:
            #app.logger.error(test_ratio+"[ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR]")
            
            schema_name = 'schema-' + str(schema_id)
            df = pd.read_sql_query('SELECT * FROM {}.{};'.format(*_ci(schema_name, table_name)), db.engine)
            app.logger.error(df.head())
            df[column_name ] = df[column_name].astype('category')
            df[column_name+"converted_cat"] = df[column_name].cat.codes

            y=df[column_name+"converted_cat"].tolist()


            df=df.drop([column_name,column_name+"converted_cat"], axis=1)
            X=df.to_numpy()

            


            #Normalizing numerical features so that each feature has mean 0 and variance 1
            feature_scaler = StandardScaler()
            X_scaled = feature_scaler.fit_transform(X)



            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=float(test_ratio), random_state=0)
            lr = Lasso(alpha = 0.05)
            lr.fit(X_train, y_train)
            pred_test_lr= lr.predict(X_test)
            

            re="Mean Squared Error == "+ str(round(mean_squared_error(y_test,pred_test_lr),4))
            re=re+ " Root MSE == "+ str(round(np.sqrt((mean_squared_error(y_test,pred_test_lr))),4))
            re=re+ " R2-score == "+ str(round(r2_score(y_test, pred_test_lr),4))
            
            return "Results: "+str(re) 
        except Exception as e:
            transaction.rollback()
            app.logger.error("[ERROR] Couldn't apply Lasso Regression on this data")
            app.logger.exception(e)
            raise e

    def svr(self, schema_id, table_name, column_name,test_ratio):
        connection = db.engine.connect()
        transaction = connection.begin()
        try:
            #app.logger.error(test_ratio+"[ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR][ERROR]")
            
            schema_name = 'schema-' + str(schema_id)
            df = pd.read_sql_query('SELECT * FROM {}.{};'.format(*_ci(schema_name, table_name)), db.engine)
            app.logger.error(df.head())
            df[column_name ] = df[column_name].astype('category')
            df[column_name+"converted_cat"] = df[column_name].cat.codes

            y=df[column_name+"converted_cat"].tolist()


            df=df.drop([column_name,column_name+"converted_cat"], axis=1)
            X=df.to_numpy()

            


            #Normalizing numerical features so that each feature has mean 0 and variance 1
            feature_scaler = StandardScaler()
            X_scaled = feature_scaler.fit_transform(X)



            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=float(test_ratio), random_state=0)
            lr = make_pipeline(StandardScaler(), SVR(C=256, epsilon=0.2))
            lr.fit(X_train, y_train)
            pred_test_lr= lr.predict(X_test)
            

            re="Mean Squared Error == "+ str(round(mean_squared_error(y_test,pred_test_lr),4))
            re=re+ " Root MSE == "+ str(round(np.sqrt((mean_squared_error(y_test,pred_test_lr))),4))
            re=re+ " R2-score == "+ str(round(r2_score(y_test, pred_test_lr),4))
            
            return "Results: "+str(re) 
        except Exception as e:
            transaction.rollback()
            app.logger.error("[ERROR] Couldn't apply SVR Regression on this data")
            app.logger.exception(e)
            raise e


class Reports:
    def __init__(self):
        pass

    def correlation_matrix(self, schema_id, table_name,):

        connection = db.engine.connect()
        transaction = connection.begin()
        try:

            schema_name = 'schema-' + str(schema_id)
            df = pd.read_sql_query('SELECT * FROM {}.{};'.format(*_ci(schema_name, table_name)), db.engine)

             

            df=df.corr()
            
            

            df=df.drop('id',1)

            corrMatrix=df.corr()

            print(corrMatrix.columns)
             

            return corrMatrix






        except Exception as e:
            transaction.rollback()
            app.logger.error("[ERROR] Couldn't compute Correlation Matrix on this data")
            app.logger.exception(e)
            raise e









class Classification:
    def __init__(self):
        pass

    def evaluation(self,x_shape,y_test,y_pred,algo_info):
        precision,recall,fscore,support=precision_recall_fscore_support(y_test, y_pred, average='macro')
        accuracy=accuracy_score(y_test, y_pred)
            

        re= "Number of mislabeled points out of a total %d points : %d" % (x_shape, (y_test != y_pred).sum())

        return [accuracy,precision,recall,fscore,re,algo_info]




    def naive_bayes(self, schema_id, table_name, column_name,test_ratio):
        connection = db.engine.connect()
        transaction = connection.begin()
        try:
             
            schema_name = 'schema-' + str(schema_id)
            df = pd.read_sql_query('SELECT * FROM {}.{};'.format(*_ci(schema_name, table_name)), db.engine)
             
            df[column_name ] = df[column_name].astype('category')
            df[column_name+"converted_cat"] = df[column_name].cat.codes

            y=df[column_name+"converted_cat"].tolist()


            df=df.drop([column_name,column_name+"converted_cat"], axis=1)
            X=df.to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_ratio), random_state=0)
            gnb = GaussianNB()
            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            #precision,recall,fscore,support=precision_recall_fscore_support(y_test, y_pred, average='macro')
            #accuracy=accuracy_score(y_test, y_pred)
            

            #re="\n"+ "Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum())


            #return "Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum())

            return  self.evaluation(X_test.shape[0],y_test,y_pred,"Naive Bayes Classfication")


 
        except Exception as e:
            transaction.rollback()
            app.logger.error("[ERROR] Couldn't apply Naive bayes on this data")
            app.logger.exception(e)
            raise e

    def k_nearest_neighbours(self, schema_id, table_name, column_name,test_ratio,k):
        connection = db.engine.connect()
        transaction = connection.begin()
        try:
            schema_name = 'schema-' + str(schema_id)
            df = pd.read_sql_query('SELECT * FROM {}.{};'.format(*_ci(schema_name, table_name)), db.engine)
             
            df[column_name ] = df[column_name].astype('category')
            df[column_name+"converted_cat"] = df[column_name].cat.codes

            y=df[column_name+"converted_cat"].tolist()


            df=df.drop([column_name,column_name+"converted_cat"], axis=1)
            X=df.to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_ratio), random_state=0)
            neigh = KNeighborsClassifier(n_neighbors=k) 
            y_pred = neigh.fit(X_train, y_train).predict(X_test)
             
            #return "Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum())

            return self.evaluation(X_test.shape[0],y_test,y_pred,"KNN")


 
        except Exception as e:
            transaction.rollback()
            app.logger.error("[ERROR] Couldn't apply KNN on this data")
            app.logger.exception(e)
            raise e


    def decision_tree(self, schema_id, table_name, column_name,test_ratio,criterion):
        connection = db.engine.connect()
        transaction = connection.begin()
        try:
            schema_name = 'schema-' + str(schema_id)
            df = pd.read_sql_query('SELECT * FROM {}.{};'.format(*_ci(schema_name, table_name)), db.engine)
            app.logger.error(df.head())
            df[column_name ] = df[column_name].astype('category')
            df[column_name+"converted_cat"] = df[column_name].cat.codes

            y=df[column_name+"converted_cat"].tolist()


            df=df.drop([column_name,column_name+"converted_cat"], axis=1)
            X=df.to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_ratio), random_state=0)
            clf = tree.DecisionTreeClassifier(criterion=criterion)
            y_pred = clf.fit(X_train, y_train).predict(X_test)
            return self.evaluation(X_test.shape[0],y_test,y_pred,"Decision Tree Classification")


 
        except Exception as e:
            transaction.rollback()
            app.logger.error("[ERROR] Couldn't apply DT on this data")
            app.logger.exception(e)
            raise e

    def artificial_neural_net(self, schema_id, table_name, column_name,test_ratio,criterion):
        connection = db.engine.connect()
        transaction = connection.begin()
        try:
             
            schema_name = 'schema-' + str(schema_id)
            df = pd.read_sql_query('SELECT * FROM {}.{};'.format(*_ci(schema_name, table_name)), db.engine)
            app.logger.error(df.head())
            df[column_name ] = df[column_name].astype('category')
            df[column_name+"converted_cat"] = df[column_name].cat.codes

            y=df[column_name+"converted_cat"].tolist()


            df=df.drop([column_name,column_name+"converted_cat"], axis=1)
            X=df.to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_ratio), random_state=0)
            clf = Perceptron()
            y_pred = clf.fit(X_train, y_train).predict(X_test)
            return self.evaluation(X_test.shape[0],y_test,y_pred,"Artificial Neural Network Classfication")


 
        except Exception as e:
            transaction.rollback()
            app.logger.error("[ERROR] Couldn't apply ANN on this data")
            app.logger.exception(e)
            raise e



    
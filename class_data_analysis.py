#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import pandas as pd
# pd.set_option()
import numpy as np
import gc

# visualizations
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set(style="whitegrid", font_scale= 1.5)
from sklearn.feature_selection import chi2



# In[70]:


class DataAnalysis():
    """
    Assign train and test attributes with pandas dataframe
    train and test data essentially have the same columns, but one extra output for train
    
    """
    def __init__(self, train, test, y):
        self.train= train 
        self.test= test
        # y is a string
        self.y= y
    
    def var_iden(self):
        """This function identifies the variables in the train data
        """
        print('*'*15, 'Variable Identification', '*'*15, '\n')
        self.columns= list(self.train.columns) 
        self.columns.remove(self.y)
        self.num_features= []
        self.cat_features= []
        for col in self.columns:
            try:
                _= self.train[col].mean()
                self.num_features.append( col )
            except:
                self.cat_features.append( col )
        
        # report 
        print('train shape: ', self.train.shape)
        print('test.shape: ', self.test.shape)
        print()
        print('Numerical features are: ')
        print( self.num_features ) 
        print()
        print('Categorical features are: ')
        print( self.cat_features )
        print()
        
        # print the description of the data
        print('description of train data: ', '\n')
        print(self.train.describe())
        print()
        print('description of test data: ', '\n')
        print(self.test.describe())
        #end
        print('#'*30, '   End   ', '#'*30, '\n\n')
        
    def na_analysis(self):
        """
        Find out the number of NaNs in different columns for both train and test data
        """
        print('*'*15, 'NAs Analysis', '*'*15, '\n')
        report= self.train.drop( columns= self.y ).isna().sum()
        print('Missing values in the train data:\n')
        print(report)
        print('')
        report= self.test.isna().sum()
        print('Missing values in the test data:\n')
        print(report)
        print('#'*30, '   End   ', '#'*30, '\n\n')
        del(report); gc.collect()
    
    def data_cleaning(self):
        """
        print a description of variables
        check for duplicates and drop the duplicated rows
        """
        print('*'*15, 'Data Cleaning', '*'*15, '\n')
        
        length= self.train.shape[0]
        length_dropped= self.train.drop_duplicates().shape[0]
        if  length!= length_dropped:
            print('There are %d duplicated rows in the train data' % 
                                    (length - length_dropped) )
            self.train= self.train.drop_duplicates()
            print('Dropped the duplicated rows in train data!')
        else:
            print('No duplicated rows in train data')
            
        length= self.test.shape[0]
        length_dropped= self.test.drop_duplicates().shape[0]
        if  length!= length_dropped:
            print('There are %d duplicated rows in the test data' % 
                                    (length - length_dropped) )
            self.test= self.test.drop_duplicates()
            print('Dropped the duplicated rows in test data!')
        else:
            print('No duplicated rows in test data')
        #end
        print('#'*30, '   End   ', '#'*30, '\n\n')    
    ###################################################
    ###############   univariate EDA  #################    
    
    def dist_cat(self, var, figsize= (8, 6)):
        """
        plots the distribution of categorical variables
        var:
            categorical column
        """
        # get the order of levels based on the counts in an ascending manner
        order= self.train[var].value_counts(ascending= True).index 
        train_percentage= self.train[var].value_counts(ascending= True)
        test_percentage= self.test[var].value_counts(ascending= True)
        data= pd.DataFrame({'train': train_percentage, 'test': test_percentage},                            columns= ['train', 'test'], index= order )

        # Since there might be level present in the train data which is not available in 
        # the test data, some NAs might get created. We simply fill those NAs with zero
        # because it can mean that level is present, and it has zero frequency. 
        data.fillna(value= 0, inplace= True)
        fig= plt.figure(figsize= figsize)
        ax= plt.axes()
        #import pdb; pdb.set_trace()
        data['train'].plot(ax= ax, kind= 'bar', color='blue', width= 0.21, position= -0.05,                            label= var + '_train', fontsize= 15 )
        #annotation_barplot(ax, TrInd_End + 1 , annotate_test= False)
        data['test'].plot(ax= ax, kind= 'bar', color='red', width= 0.2, position= 1.05,                           label= var + '_test', fontsize= 15 )

        #annotation_barplot(ax, TeInd_End - 1, annotate_test= True)
        ax.set_xlabel(var, fontsize= 15 )
        ax.set_ylabel('Counts', fontsize= 15 )
        ax.legend() 

    def dist_num(self, var, figsize= (8, 6)):
        """
        plots the distribution of numerical variables
        var:
            numerical column
        """
        fig= plt.figure(figsize= figsize )
        ax= plt.axes()

        ax.set_xlabel( var, fontsize= 15)
        ax.set_ylabel("Distribution", fontsize= 15)
        ax.tick_params(labelsize= 15)
        
        # for categorical variables, NaNs are automatically dropped when obtaining 
        # value_counts. But for numerical features, we must drop them manually!
        sns.distplot(self.train[var].dropna(), label= var + '_train')
        sns.distplot(self.test[var].dropna(), label= var + '_test' )

        plt.legend(fontsize= 15)
    
    def dispersion_num(self, var, figsize= (8,4)):
        """
        plots the dispersion of a numerical variable w.r.t row index
        Good for discovering irregularities, if any exists...
        """
    
        fig= plt.figure(figsize= figsize )
        plt.xlabel("row", fontsize= 15)
        plt.ylabel( var, fontsize= 15)
        plt.tick_params(labelsize= 15)
        combined= pd.concat( [self.train, self.test], sort= False )
        combined.index= range(len(combined))
        plt.plot(combined[var], '.' )
        try:
            plt.axvline(x= len(self.train), color= 'red')
        except:
            import pdb; pdb.set_trace()
        del(combined); gc.collect()
    ###################################################
    ###############   bivariate EDA  ##################
    def correlation_matrix(self, df, figsize= (20, 20)):
        """
        For a dataframe, it returns a heatmap of all the correlations between numerical 
        variables
        
        It is better to analyze this for the train data
        """
        corr= df.corr()

        f, ax = plt.subplots(figsize= figsize)
        plt.tick_params(labelsize= 15)

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap= True)
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask= mask, cmap= cmap, vmax= 1, vmin= -1, annot= True,
                center= 0,  square= True, linewidths= 0.5, linecolor= 'black' , 
                    cbar_kws= {"shrink": .5})
    #########################################################################
    ###############   chi-2 test for categorical features  ##################

    def chi2_heatmap(self, cat_features, figsize= (10,10), alpha= 0.05):
        """
        It returns the p_value between every pair of categorical variables in a heatmap
        and masks thos p_values which are greather than p_val with red cells 
        
        * important point: Since one of our goals is to find out the dependency 
            between the response variable and other features, we perform the chi-2
            test only on the train data

        Arguments:
            - cat_features: all categorical features. Add extra if needed.
            - figsize
            - alpha: the threshold to which the p_values are compared. p_values 
                greather than alpha will be masked.
        
        """
        
        # each time, one feature is assume to be the respnse variable and its dependency with other
        # features is analyzed
        table= pd.DataFrame(columns= cat_features, index= cat_features)
        for el in cat_features:
            X= self.train[ list(cat_features - {el}) ]
            y= self.train[el]
            chi_scores= chi2(X, y)
            p_values = pd.Series(chi_scores[1], index = X.columns)
            table[el]= p_values
            #import pdb; pdb.set_trace()
            
        f, ax = plt.subplots(figsize= figsize)
        plt.tick_params(labelsize= 15)

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap= True)
        #mask = np.zeros_like(table, dtype= np.bool)
        #mask[np.triu_indices_from(mask)] = True
        mask= table[table > 0.05].fillna(0).astype(bool)
      
        # Draw the heatmap with the mask and correct aspect ratio
        ax= sns.heatmap(table, mask= mask, cmap= cmap, vmax= 1, vmin= -1, annot= True,
                    center= 0,  square= True, linewidths= 0.5, linecolor= 'black' , 
                    cbar_kws= {"shrink": .5})
        ax.set_facecolor("red")
        ax.set_xlabel('Considered as response variable', fontsize= 15)
        return table

# In[ ]:





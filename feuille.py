#!/usr/bin/env python
# coding: utf-8

# In[17]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.regression import LinearRegression
import pandas as pd


# In[18]:


spark=SparkSession.builder.appName('projet').getOrCreate()


# In[4]:


data_train=spark.read.csv("data_train.csv",inferSchema=True ,header=True)
data_text=spark.read.csv("test_data.csv",inferSchema=True ,header=True)


# In[5]:


indexer = StringIndexer(inputCol="type", outputCol="type_index").fit(data_train)
indexer2 = StringIndexer(inputCol="type", outputCol="type_index").fit(data_text)


# In[6]:


df_train = indexer.transform(data_train)
df_test = indexer2.transform(data_text)


# In[16]:


out=df_train.toPandas()
out


# In[19]:


df_train.dtypes




# In[13]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[14]:


featureassembler=VectorAssembler(inputCols=["a","area","ci","pi","eccentricity","kx","ky","m00","m01","m10","minAreaPercent","minEnclosingCircleArea","mu02","mu03","mu11","mu20","mu30","sx","sy","d"],outputCol="features")


# In[20]:


output=featureassembler.transform(df_train)


# In[21]:


output.select("features").show(5)


# In[344]:


output.columns


# In[24]:


finalized_data=output.select("features","type_index","type","d")
finalized_data.show(5)


# In[25]:


#train_data,test_data=finalized_data.randomSplit([0.90,0.10])


# In[36]:


regressor=LinearRegression(featuresCol= 'features', labelCol='type_index',maxIter=100, elasticNetParam=0.8)
regressor=regressor.fit(finalized_data)


# In[37]:


regressor.coefficients


# In[38]:


import matplotlib.pyplot as plt
import numpy as np
beta = np.sort(regressor.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()


# In[39]:


regressor.intercept


# In[40]:


trainingSummary = regressor.summary
print("numIterations: %d" % trainingSummary.totalIterations)


# In[46]:


test_data=featureassembler.transform(df_test)
rest=regressor.transform(test_data)
df=rest.toPandas()
rest.select("type","type_index","prediction").show(6)


# In[42]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator=BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='type_index')


# In[44]:



print("The area under ROC for test set is {}".format(evaluator.evaluate(rest)))


# In[ ]:





# In[47]:



#import data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)


# In[49]:


g = sns.lmplot(x="prediction", y="type_index",data=df)


# In[51]:


g = sns.lmplot(x="prediction", y="type_index",col="type", hue="type",col_wrap=4,height=3,data=df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





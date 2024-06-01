# Background and Purpose

Ames, Iowa is the college town of Iowa State University. The Ames dataset consists of the housing sale records between 2006-2010, including features like their attributes and sale prices. 
The goal of this project is to provide descriptive models of the key features, and then develop a machine learning algorithm for future housing sale prices, all backed by data analytics. The aim is for high accuracy and for high variance (R-Squared).
Ideally, one would find the most significant contributing factors using multiple linear regression, then one would find the most important indicators using random forest.

# Business Applications

Data analytics and machine learning are critical to modern businesses. Data analytics allow for businesses to gain insights in order to guide and/or enhance their operations, while machine learning methods can reduce risk when making investments/decisions. 
The housing/sales model is an example, and this structure (data analytics, finding best predictors/most significant factors, and creating a predictive model) can be applied to other business cases. In the presented case, numerous methods were used during the data analytics and even a predictive process, such as a time series, which can be transferred almost on a one-to-one basis with other cases. In fact, it was helpful that a financial "target" was used, therefore giving the analysis more applications to the business world.

# Future Works

The point of the project was two-fold: create a descriptive model and also to create a predictive model. A certain number of methods were used for both, but many other methods could have been used instead.
For example, descriptive models include feature selection, feature engineering, multiple linear regression, penalized linear regression, stepwise regression (AIC, BIC, etc.), Support Vector Regression, Random Forest, Gradient Boosting, and others.
Predictive models include gridsearchCV, and can also use multiple/penalized linear regression, gridsearch, Random Forest, Gradient Boosting, and other models. A more accurate model can always be created, yet the most accurate model is not always optimal for the purposes of improving business efficiency. We know from our model that the random forest was not optimal, and it's recommended to have better tuning. 


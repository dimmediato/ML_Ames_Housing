# Background and Purpose

Ames, Iowa is the college town of Iowa State University. The Ames dataset consists of the housing sale records between 2006-2010, including features like their attributes and sale prices. 
The goal of this project is to provide descriptive models of the key features, and then develop a machine learning algorithm for future housing sale prices, all backed by data analytics. The aim is for high accuracy and for high variance (R-Squared).
Ideally, one would find the machine learning model with the highest cross-validation R-Squared, and compare its positives and negatives with the other models.

# Business Applications

Data analytics and machine learning are critical to modern businesses. Data analytics allow for businesses to gain insights in order to guide and/or enhance their operations, while machine learning methods can reduce risk when making investments/decisions. 
The housing/sales model is an example, and this structure (data analytics, finding best predictors/most significant factors, and creating a predictive model) can be applied to other business cases. In the presented case, numerous methods were used during the data analytics and even a predictive process, which can be transferred almost on a one-to-one basis with other cases. In fact, it was helpful that a financial "target" was used, therefore giving the analysis more applications to the business world.

# Future Works

The point of the project was two-fold: create a descriptive model and also to create a predictive model. A certain number of methods were used for both, but many other methods could have been used instead. This includes different types of models, such as XGBooster, using feature selection via the lasso regressions, or modifying the parameters to better suite our needs. Another note is that the focus of the project was on the CV score of the models, but not necessarily what the models can actually produce themselves (such as predictions, coefficients, and correlations). Furthermore, the pipelines themselves can always be improved, such as creating new features to make up for inadequate/redundant features in the model, such as a TotalBaths feature to deal with all bathroom features.


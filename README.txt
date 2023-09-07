# kaggleWaterImpute

This is a small project relating to this challenge https://www.kaggle.com/competitions/playground-series-s3e21/overview
Using the training data from here : https://www.kaggle.com/datasets/vbmokin/dissolved-oxygen-prediction-in-river-water

I wanted to impute missing values, while capturing the uncertainty of imputed values. I created a distribution of possible values using
the jackknife plus method, and I imputed that into copies of the rows, the idea is roughly shown in imputation.png. I don't know if this has an offical name.
Unfortunately this method wasn't useful in this particular challenge but I think it is interesting.

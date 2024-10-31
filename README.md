# Data Cleaning

Remove missing data

Remove duplications

Catch suspicious data

Fix or Clean suspicious data


# Feature Engineering

Creat new features aiming at reducing covariance

Convert a numerical response variable to a categorical response variable


Before:

![image](https://github.com/user-attachments/assets/79b2f825-7191-4c6e-be86-0c1e02da14dd)

After:

![image](https://github.com/user-attachments/assets/3c1c1285-93cc-44c1-bf83-2bdddda8ac52)

C-Q plot comparison:

![image](https://github.com/user-attachments/assets/5e5a1fa8-857d-4bae-96b0-ae9a982bb51a)


# Classification: determine the price is high or low

Logistic Regression Accuracy: 94%

Initial Support Vector Machine Accuracy: 91%

Support Vector Machine Accuracy After Hyperparameter Tuning: 95%

Single Decision Tree Accuracy: 90%

Random Forest: 95%

![image](https://github.com/user-attachments/assets/43127091-0ec0-4950-b14c-f018e75c8608)

![image](https://github.com/user-attachments/assets/324bdbc9-a7e4-4641-82ab-065c8c2b37bf)

The random forest has the highest accuracy (95.48%), suggesting it is the most effective model for this classification task. It’s robust and captures complex patterns, making it suitable for production use. 

Random forests reduce the risk of overfitting by averaging across multiple trees, and they capture non-linear relationships better than logistic regression and individual decision trees. 

They also provide insights into feature importance.


# Regression: predict the price

Support Vector Machine MSE: 404796.3

![image](https://github.com/user-attachments/assets/9ad96fbd-81ed-4cb2-bc89-4711a146598e)

The model is fairly accurate for lower and mid-range prices but shows more variability or error at the higher price range.

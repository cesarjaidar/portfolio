---
title: "Feature Selection and Dimensionality Reduction in Linear Regression and Classification"
excerpt: "Dimensionality reduction techniques such as PCA, variance thresholding, and feature selection are applied to housing price prediction and mushroom classification datasets to improve model efficiency and interpretability."
collection: portfolio
---

Description:

This project explores feature selection and dimensionality reduction techniques in two parts:

Linear Regression on Housing Data

🏡 PCA and variance-based feature selection are applied to predict house sale prices.

📉 Models are evaluated using R2 and RMSE to compare the effectiveness of PCA versus high variance feature selection.

PCA retained 90% variance and achieved an R2 of 0.842, outperforming the high variance method (R2 of 0.655).

Classification on Categorical Mushroom Data

🍄 A decision tree classifier is used to predict whether mushrooms are poisonous or edible.

📊 Feature selection using χ2-statistic identified the top 5 features, leading to a slight reduction in accuracy (from 100% to 92.49%).

Visualization of the decision tree provides insights into key decision paths.

This project demonstrates the trade-offs between dimensionality reduction, model accuracy, and interpretability across regression and classification tasks.


🔗 [View Project on GitHub](https://github.com/cesarjaidar/portfolio/blob/master/files/Feature%20Selection%20and%20Dimensionality%20Reduction.py)

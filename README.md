# Restaurant-Recommender
## Yelp Recommender System (DSCI 553 Final Competition Project)

## Overview
- This project builds a model-based recommendation system to predict Yelp user ratings for businesses.
- It was developed for the DSCI 553: Foundations & Applications of Data Mining course competition, which required achieving     the lowest RMSE on a hidden test set under strict runtime and resource constraints.

## Results
- I combined PySpark for distributed feature engineering and XGBoost for regression modeling, achieving 0.973 RMSE on the final hidden test set.

## Quick Implementation Note:
- This implementation was optimized for a competition setting with strict runtime limits, so some parts (like nested tuple indexing in PySpark) are intentionally unrefined for speed. In a production setting, I would refactor it into modular, maintainable code with clearer data schemas.
- 
## Technical Approach
- End-to-end ML pipeline: data preprocessing, feature engineering, model tuning, and evaluation
- Feature-rich approach: engineered business-, user-, and interaction-level features from multiple JSON and CSV sources
- Scalable data processing: used PySpark to join and aggregate large Yelp datasets efficiently
- Model optimization: tuned XGBoost hyperparameters using RandomizedSearchCV followed by GridSearchCV
- Performance-driven: validated improvements iteratively, balancing feature richness with runtime
- Key takeaway: thoughtful feature engineering improved performance more than model complexity alone

## Model choice
- Selected XGBRegressor for its balance of speed and accuracy
- Tuned hyperparameters locally using RandomizedSearchCV followed by GridSearchCV
- Tested more complex approaches (e.g., clustering + ensemble models) that were too slow on this dataset and did not improve RMSE within time limits
- Initially experimented with collaborative filtering and a hybrid weighted-average approach combining collaborative filtering and XGBoost, but found that optimizing XGBoost alone yielded faster and better results for this competition

## Evaluation
- Computed RMSE and error distribution:
  >=0 and <1   n = 102,326  
  >=1 and <2   n = 32,796  
  >=2 and <3   n = 6,121  
  >=3 and <4   n = 801  
  >=4          n = 0  
  Runtime measured on Vocareum: ~10 minutes

## Results:
- Validation RMSE: 0.977
- Final Test RMSE: 0.973
- Runtime: ~594 seconds

## Tech stack
- PySpark for distributed data processing
- XGBoost for regression modeling
- NumPy, statistics, and collections for feature manipulation

## Future improvements and production considerations
- Refactor into modular, production-ready scripts for preprocessing, training, and evaluation
- Optimize Spark usage by reducing collect() calls and replacing nested tuple indexing with clearer structured joins or         DataFrame schemas
- Further experiment with hybrid approaches such as clustering (e.g., DBSCAN) or matrix factorization combined with XGBoost
- If deployed as an online algorithm, implement user/item statistical features updated over time—this approach showed strong    RMSE reduction when sequential data was available but was less effective here due to random test data
- Conduct A/B tests for model updates and new feature sets in a live recommendation setting
- Explore online learning or incremental updates to improve personalization over time

from helper import *

if __name__ == '__main__':
	data1 = load_data('./data/World Happiness Report 2021.csv')
	data2 = load_data('./data/World Happiness Report 2022.csv')
	formatted_data1 = format_train_data1(data1)
	formatted_data2 = format_train_data2(data2)
	cleaned_data1 = clean_data(formatted_data1)
	cleaned_data2 = clean_data(formatted_data2)

	### Fitting Regular Machine Learning Models
	X, Y, feat_col = extract_scale_features(cleaned_data1, cleaned_data2, False, False)
	# fit_data(X, Y, feat_col)

	### Graphing Functions
	#graph_MSER2()
	#graph_mses()

	### Fitting Neural Network
	# X, Y, feat_col = extract_scale_features(cleaned_data1, cleaned_data2, False, True)
	# nn_fit_data(X, Y)

	### Fitting Regular Machine Learnings Models with Feature Engineering
	# X, Y, feat_col = extract_scale_features(cleaned_data1, cleaned_data2, True, False)
	# fit_data(X, Y, feat_col)
	
    # Optimization Steps
    # 1. Larger training and testing sets (i.e., combine years or use different years for each)
		# No significant change in the R2 score or MSE.
		# When we combine the 2021 and 2022 datasets, our R2 scores and MSE improve across every model.
    # 2. Hyperparameter tuning
		# Use RandomizedSearchCV() to find the best hyperparameters.
		# Implicity employs cross-validaiton as well.
		# Best Params: kernel: rbf, gamma: scale or 0.1, C: 1
		# Epsilon (margin width) seems to improve performance very slightly because the narrower
		# margin width precludes unimportant support vectors that would have influenced the decision boundary poorly.
    # 3. Ensemble learning
		# Random Forest Regressor (bagging) preforms nearly the same, if not the exact same, as SVR.
			# Insignificant change in peformance after hyperparameter tuning.
			# Both cases could be because the SVR is already optmized the best it can be.
		# Gradient Boosting Regressor preforms the worst, so hyperparameter tuning needed.
			# After tuning, RandomizedSearchCV recommends a slow learning rate and 300 trees, which is 
			# decently higher than the default 100. Correlation is that a lower learning rate requires
			# more trees because there will be more iterations.
			# Preforms worse than all other models!
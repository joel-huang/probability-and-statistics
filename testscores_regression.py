import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

r2_scores = []
midterm = pd.read_csv('scores.csv', usecols=[8])

def print_regression(dat, pre):
    plt.scatter(dat.values, midterm.values)
    plt.plot(dat.values, pre)
    plt.xlabel('past score')
    plt.ylabel('predicted midterm score')
    plt.show()

for i in range(0,8):
    data = pd.read_csv('scores.csv', usecols=[i])
    
    linreg = LinearRegression()
    linreg.fit(X=data.values, y=midterm.values)

    # predicted midterm values
    pred = linreg.predict(data.values)

    # store r2 scores in array
    r2_scores.append(r2_score(data.values, pred))

# find the best r2 score and record its index
# assume no repetition of r2 score for convenience
best_index = (r2_scores.index(max(r2_scores)))
best_data = pd.read_csv('scores.csv', usecols=[best_index])
best_linreg = LinearRegression().fit(X=best_data.values, y=midterm.values)
best_pred = best_linreg.predict(best_data.values)

# print results
print("Best prediction is column " + str(best_index) + " with r2 score: " + str(r2_scores[best_index]))
print_regression(best_data, best_pred)
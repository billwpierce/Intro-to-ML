# ## Logistic Regression
# from Carl Shan and Jen Selby
# 

# ### Step 1: Importing the libraries we need
# Remember that you can run this cell with SHIFT+ENTER
import numpy.random
from sklearn import linear_model
import matplotlib.pyplot

################################################################################
#  GENERATED DATA
################################################################################

# We have two types of plants
# Plant A tends to be taller (avg 60cm) and thinner (avg 8cm)
# Plant B tends to be shorter (avg 58cm) and wider (avg 10cm)
# We'll use 0 for Plant A and 1 for Plant B

NUM_INPUTS = 50
PLANT_A_AVG_HEIGHT = 60.0
PLANT_A_AVG_WIDTH = 8.0
PLANT_B_AVG_HEIGHT = 58.0
PLANT_B_AVG_WIDTH = 10.0

plantA_heights = numpy.random.normal(loc=PLANT_A_AVG_HEIGHT, size=NUM_INPUTS)
plantA_widths = numpy.random.normal(loc=PLANT_A_AVG_WIDTH, size=NUM_INPUTS)

plantB_heights = numpy.random.normal(loc=PLANT_B_AVG_HEIGHT, size=NUM_INPUTS)
plantB_widths = numpy.random.normal(loc=PLANT_B_AVG_WIDTH, size=NUM_INPUTS)

plant_inputs = list(zip(numpy.append(plantA_heights, plantB_heights), numpy.append(plantA_widths, plantB_widths)))

print(plant_inputs)

types = [0]*NUM_INPUTS + [1]*NUM_INPUTS

print(types)


################################################################################
# PLOT
################################################################################

# put the generated points on the graph
matplotlib.pyplot.scatter(plantA_heights, plantA_widths, c="red", marker="o")
matplotlib.pyplot.scatter(plantB_heights, plantB_widths, c="blue", marker="^")

################################################################################
# MODEL TRAINING
################################################################################

model = linear_model.LogisticRegression()
model.fit(plant_inputs, types)

print('Intercept: {0}  Coefficients: {1}'.format(model.intercept_, model.coef_))

################################################################################
# PREDICTION
################################################################################

# Generate two new plants, one for A and one for B.

newA_height = numpy.random.normal(loc=PLANT_A_AVG_HEIGHT)
newA_width = numpy.random.normal(loc=PLANT_A_AVG_WIDTH)
newB_height = numpy.random.normal(loc=PLANT_B_AVG_HEIGHT)
newB_width = numpy.random.normal(loc=PLANT_B_AVG_WIDTH)

# turns new points into an array

inputs = [[newA_height, newA_width], [newB_height, newB_width]]

# Predicts the chance ofthose points fitting in with the current type.

print('Type predictions: {0}'.format(model.predict(inputs)))
print('Probabilities: {0}'.format(model.predict_proba(inputs)))

# Marks Point A
if model.predict_proba(inputs)[0][0] > model.predict_proba(inputs)[0][1]:
	matplotlib.pyplot.scatter(inputs[0][0], inputs[0][1], c="orange", marker="o")
else:
	matplotlib.pyplot.scatter(inputs[0][0], inputs[0][1], c="green", marker="^")

#Marks Point B
if model.predict_proba(inputs)[1][0] > model.predict_proba(inputs)[1][1]:
	matplotlib.pyplot.scatter(inputs[1][0], inputs[1][1], c="orange", marker="o")
else:
	matplotlib.pyplot.scatter(inputs[1][0], inputs[1][1], c="green", marker="^")

matplotlib.pyplot.show()


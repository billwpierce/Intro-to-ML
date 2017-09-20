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
# Plant C tends to be shorter (avg 50cm) and thinner (avg 5cm)

NUM_INPUTS = 50
PLANT_A_AVG_HEIGHT = 60.0
PLANT_A_AVG_WIDTH = 8.0
PLANT_B_AVG_HEIGHT = 58.0
PLANT_B_AVG_WIDTH = 10.0
PLANT_C_AVG_HEIGHT = 50.0
PLANT_C_AVG_WIDTH = 5.0

plantA_heights = numpy.random.normal(loc=PLANT_A_AVG_HEIGHT, size=NUM_INPUTS)
plantA_widths = numpy.random.normal(loc=PLANT_A_AVG_WIDTH, size=NUM_INPUTS)

plantB_heights = numpy.random.normal(loc=PLANT_B_AVG_HEIGHT, size=NUM_INPUTS)
plantB_widths = numpy.random.normal(loc=PLANT_B_AVG_WIDTH, size=NUM_INPUTS)

plantC_heights = numpy.random.normal(loc=PLANT_C_AVG_HEIGHT, size=NUM_INPUTS)
plantC_widths = numpy.random.normal(loc=PLANT_C_AVG_WIDTH, size=NUM_INPUTS)

plant_inputs = list(zip(numpy.append(plantA_heights, plantB_heights), numpy.append(plantA_widths, plantB_widths)))
plantC_inputs = list(zip(plantC_heights, plantC_widths))
plant_inputs = list(numpy.append(plant_inputs, plantC_inputs))

################################################################################
# MODEL TRAINING
################################################################################

#Generate types for the 3 different logistic models
plantA_types = [0]*NUM_INPUTS + [1]*(NUM_INPUTS*2)
plantB_types = [1]*NUM_INPUTS + [0]*NUM_INPUTS + [1]*NUM_INPUTS
plantC_types = [1]*(NUM_INPUTS*2) + [0]*NUM_INPUTS

modelA = linear_model.LogisticRegression()
modelA.fit(plant_inputs, plantA_types)

modelB = linear_model.LogisticRegression()
modelB.fit(plant_inputs, plantB_types)

modelC = linear_model.LogisticRegression()
modelC.fit(plant_inputs, plantC_types)

################################################################################
# PREDICTION
################################################################################
################################################################################
# PLOT
################################################################################

# put the generated points on the graph
matplotlib.pyplot.scatter(plantA_heights, plantA_widths, c="red", marker="o")
matplotlib.pyplot.scatter(plantB_heights, plantB_widths, c="blue", marker="^")
matplotlib.pyplot.scatter(plantC_heights, plantC_widths, c="yellow", marker="s")

matplotlib.pyplot.show()


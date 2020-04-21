# Script to find micro and macro F1 scores given two input files 
import sys
import string

# Calculates precision metric
def precision(A,B):
	return A / (A + B)

# Calculates recall metric
def recall(A, C):
	return A / (A + C)

# Calculates F1 metric from precision and recall
def F1(P, R):
	return (2 * P * R) / (P + R)

if len(sys.argv) != 3:
	print("Usage: python3 calcF1.py [classifier_output] [true_output]")
	exit(-1)

# Dictionary to hold the classes and their info
# Format:
#	classes_dict["class"]["py_ay"] = # of values predicted to be this class that actually are this class
#	classes_dict["class"]["py_an"] = # of values predicted to be this class that actually are NOT this class
# 	classes_dict["class"]["pn_ay"] = # of values predicted to NOT be this class that actually are this class
# 	classes_dict["class"]["pn_pn"] = # of values predicted to NOT be this class that actually are NOT this class
classes_dict = dict()

# Get classes from the true input file
true_file = open(sys.argv[2], "r")
true_lines = []
for line in true_file:
	cl = line.strip()
	true_lines.append(cl)

	if cl not in classes_dict.keys():
		classes_dict[cl] = {}
		classes_dict[cl]["py_ay"] = 0
		classes_dict[cl]["py_an"] = 0
		classes_dict[cl]["pn_ay"] = 0
		classes_dict[cl]["pn_an"] = 0

# Get lines in the classifier file
class_file = open(sys.argv[1], "r")
class_lines = [line.strip() for line in class_file]

if len(class_lines) != len(true_lines):
	print("Error: Input files do not have same amount of results!")
	exit(-1)

# Populate the classes_dict with relevant info
for class_line, true_line in zip(class_lines, true_lines):
	if class_line == true_line:
		classes_dict[true_line]["py_ay"] += 1
	else:
		classes_dict[class_line]["py_an"] += 1
		classes_dict[true_line]["pn_ay"] += 1
		
		# For every class that was correctly NOT predicted
		for key in classes_dict.keys():
			if (key != class_line) and (key != true_line):
				classes_dict[key]["pn_an"] += 1


# Micro F1: Treat all decisions as equally weighted
A_micro = sum(classes_dict[c]["py_ay"] for c in classes_dict)
B_micro = sum(classes_dict[c]["py_an"] for c in classes_dict)
C_micro = sum(classes_dict[c]["pn_ay"] for c in classes_dict)
D_micro = sum(classes_dict[c]["pn_an"] for c in classes_dict)
p_micro = precision(A_micro, B_micro)
r_micro = recall(A_micro, C_micro)
F1_micro = F1(p_micro, r_micro)


# Macro F1: Treat all categories as equally weighted
p_macro = sum( precision(classes_dict[c]["py_ay"],classes_dict[c]["py_an"]) for c in classes_dict ) / len(classes_dict)
r_macro = sum( recall(classes_dict[c]["py_ay"],classes_dict[c]["pn_ay"]) for c in classes_dict ) / len(classes_dict)
F1_macro = F1(p_macro, r_macro)

print("Micro F1: " + str(F1_micro))
print("Macro F1: " + str(F1_macro))


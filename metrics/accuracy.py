# Script to find the accuracy given two input files 
import sys
import string

if len(sys.argv) != 3:
	print("Usage: python3 accuracy.py [classifier_output] [true_output]")
	exit(-1)


class_file = open(sys.argv[1], "r")
class_lines = [(line.strip()).split() for line in class_file]

true_file = open(sys.argv[2], "r")
true_lines = [(line.strip()).split() for line in true_file]

if len(class_lines) != len(true_lines):
	print("Error: Input files do not have same amount of results!")
	exit(-1)

num_results = len(class_lines)

num_correct = 0

for class_line, true_line in zip(class_lines, true_lines):
	if class_line == true_line:
		num_correct += 1

accuracy = num_correct / num_results

print("CORRECT: " + str(num_correct))
print("INCORRECT: " + str(num_results - num_correct))
print("ACCURACY: " + str(accuracy))	

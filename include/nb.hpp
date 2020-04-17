/* Helper declarations for Naive Bayes implementation */
#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>


/* Splits string into tokens */
std::vector<std::string> tokenize(std::string s);


/* 
	Finds log prior probability of the category (Solely based on training data)
	
	categ_freq = Number of docs that were the specific category 
	num_docs = Total Number of docs

*/
float log_prior_prob(int categ_freq, int num_docs);


/* 
	Obtain log prob with Laplace smoothing for a term in the category 

	categ_term_count = Number of times the term has appeared in the current class (Comes from training)
	categ_total_terms = Total number of terms in the current class (Comes from training)
*/
float log_term_categ_prob(float categ_term_count, float categ_total_terms);

/* 
	Performs training of the classifier.
	Reads file and computes statistics for tokens.
	Most of the variables that get passed are passed by reference so this function can modify them
	and they can be used in other functions, rather than declaring them as global.
*/
void train(std::string filename, 
				int &num_training_docs, 
				std::vector<std::string> &term_vec, 
				std::vector<std::string> &classes_vec,
				std::map<std::string,std::map<std::string,int> > &term_class_freq_map, 
				std::map<std::string,std::vector<int> > &class_info_map);

/* 
	Performs learning, which is just computing the probabilities of the terms (including Laplace Smoothing).
	Returns a (# of Terms x # of Classes) matrix containing the probabilites. 
*/
float * learn(std::vector<std::string> &term_vec, 
				std::vector<std::string> &classes_vec, 
				std::map<std::string,std::map<std::string,int> > &term_class_freq_map, 
				std::map<std::string,std::vector<int> > &class_info_map);

/*
	Performs the actual classifying of test documents.
	Returns a vector containing the index predicted class (which can be used to find the string in classes_vec)
*/
std::vector<int> test(std::string filename,
						int &num_training_docs, 
						float *prob_matrix, 
						std::vector<std::string> &term_vec, 
						std::vector<std::string> &classes_vec,
						std::map<std::string,std::vector<int> > &class_info_map);



/* CPU Implementation of the Naive Bayes Classifier */
#include "./include/nb.hpp"

int main(int argc, char **argv)
{
	if(argc != 4)
	{
		std::cerr << "Usage: " << argv[0] << " [train_file] [test_file] [output_file]" << std::endl;
		exit(-1);
	}

	std::string train_filename = argv[1]; 
	std::string test_filename = argv[2];
	std::string output_filename = argv[3];


	/*
	********************** 
	       Training 
	**********************
	*/
	/* Holds the number of training docs (Will be incremented in train() and usable in other functions) */
	int num_training_docs = 0;

	/* Store training terms in a vector */
	std::vector<std::string> term_vec;
	
	/* Store classes in a vector */
	std::vector<std::string> classes_vec;

	/* Map of term to class_list map so we can count the number of times the term is in the class */
	std::map<std::string, std::map<std::string, int> > term_class_freq_map;

	/* 
		Map of class to num_docs,num_terms 
		num_docs = number of docs of a certain class ([0])
		num_terms = number of terms in the class ([1])
	*/
	std::map<std::string, std::vector<int> > class_info_map;  

	std::cerr << "Started training" << std::endl;
	/* Perform the training (which will update all the vectors) */
	train(train_filename, num_training_docs, term_vec, classes_vec, term_class_freq_map, class_info_map);



	/*
	**********************
	       Learning
	**********************
	*/
	/* Populate a prob matrix using the vectors with training statistics */
	std::cerr << "Started Learning" << std::endl;
	float *prob_matrix = learn(term_vec, classes_vec, term_class_freq_map, class_info_map);



	/*
	*********************
	       Testing
	*********************
	*/
	std::cerr << "Started Testing" << std::endl;
	std::vector<int> results = test(test_filename, num_training_docs, prob_matrix, term_vec, classes_vec, class_info_map);



	/*
	*****************************
	       Writing to File
	*****************************
	*/
	std::ofstream outfile(output_filename);
	for(int i = 0; i < results.size(); i++)
	{
		/* Convert result into class string */
		std::string test_doc_class = classes_vec[results[i]];
		outfile << test_doc_class << std::endl;
	}

	outfile.close();

	std::cout << "Predictions can be found in " << output_filename << std::endl;

}







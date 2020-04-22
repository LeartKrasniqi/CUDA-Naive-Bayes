/* CPU Implementation of the Naive Bayes Classifier */
#include "./include/nb.hpp"
#include <chrono>
#define DEBUG 0
#define timeNow() std::chrono::high_resolution_clock::now()
#define duration(start, stop) std::chrono::duration_cast<std::chrono::seconds>(stop - start).count()

typedef std::chrono::high_resolution_clock::time_point TimeVar;    

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

	std::cerr << "Started training... ";
	TimeVar train_start = timeNow();

	/* Perform the training (which will update all the vectors) */
	train(train_filename, num_training_docs, term_vec, classes_vec, term_class_freq_map, class_info_map);
	
	TimeVar train_stop = timeNow();
	std::cerr << "Done (" << duration(train_start, train_stop) << " s)" << std::endl;



	/*
	**********************
	       Learning
	**********************
	*/
	/* Populate a prob matrix using the vectors with training statistics */
	std::cerr << "Started Learning... ";
	TimeVar learn_start = timeNow();
	float *prob_matrix = learn(term_vec, classes_vec, term_class_freq_map, class_info_map);
	TimeVar learn_stop = timeNow();
	std::cerr << "Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(learn_stop - learn_start).count() << " ms)" << std::endl;
	


	/*
	*********************
	       Testing
	*********************
	*/
	std::cerr << "Started Testing... "; 
	TimeVar test_start = timeNow();
	std::vector<int> results = test(test_filename, num_training_docs, prob_matrix, term_vec, classes_vec, class_info_map);
	TimeVar test_stop = timeNow();
	std::cerr << "Done (" << duration(test_start, test_stop) << " s)" << std::endl;



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

	std::cout << "\nPredictions can be found in " << output_filename << std::endl;



	/*
	*******************
	       DEBUG
	*******************
	*/
	#if DEBUG

	std::ofstream debugfile("prob_matrix.txt");

	for(int c_idx = 0; c_idx < classes_vec.size(); c_idx++)
		debugfile << classes_vec[c_idx] << " ";
	debugfile << std::endl << std::endl;

	for(int t_idx = 0; t_idx < term_vec.size(); t_idx++)
	{
		std::string term = term_vec[t_idx];
		debugfile << term << "\t\t\t";
		for(int c_idx = 0; c_idx < classes_vec.size(); c_idx++ )
		{
			float prob = *(prob_matrix + (t_idx * classes_vec.size()) + c_idx);
			debugfile << prob << " ";
		}
		debugfile << std::endl;
	}
	debugfile.close();
	#endif

}







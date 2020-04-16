#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <string>
#include <sstream>


/* Function to convert vector of ints into array of ints */
int * vecToArr(std::vector<int> v)
{
	int *arr = (int *)malloc(v.size() * sizeof(int));
	if(arr == NULL)
	{
		std::cerr << "Error converting vector to array" << std::endl;
		exit(-1);
	}

	std::copy(v.begin(), v.end(), arr);

	return arr;
}


int main(int argc, char **argv)
{
	if(argc != 3)
	{
		std::cerr << "Usage: " << argv[0] << " [train_file] [test_file]" << std::endl;
		exit(-1);
	}

	/* Use vector to store terms */
	std::vector<std::string> term_vec;
	int term_index = 0;

	/* Map of term to document list, to make sure no duplicate documents are added to list */
	std::map<std::string, std::vector<int> > term_doc_map;

	/* 
		Vector of terms.  
		Each index represents the term.
		The value at that index represents the index in doc_term that holds list of documents for the term
		Note: Will be converted to array later (to be used in kernel function)
	*/
	std::vector<int> term_index_vec;


	/* 
		Vector of documents.
		Each value represents the doc_number that the term has appeared in
		Note: Will be converted to array later (to be used in kernel function)
	*/
	std::vector<int> doc_term_vec;

	/* Vector to hold all the classes */
	std::vector<std::string> classes_vec;

	/* Loop through each document */
	std::ifstream file(argv[1]);
	std::string line;
	int lineno = 0; 
	while (std::getline(file, line)) 
	{
		/* 
			Split string 
			doc_split[0] = doc_class
			doc_split[1 -> end] = terms in doc
		*/
		std::istringstream iss(line);
		std::vector<std::string> doc_split((std::istream_iterator<std::string>(iss)),
                                 std::istream_iterator<std::string>());
		//doc_split.push_back(std::to_string(lineno));

		/* Append class to classes_vec, only if it has not been seen before */
		std::vector<std::string>::iterator class_it = std::find(classes_vec.begin(), classes_vec.end(), doc_split[0]);
        if(class_it == classes_vec.end())
            classes_vec.push_back(doc_split[0]);

        /* Loop through each term in the document */
        for(int i = 1; i < doc_split.size(); i++)
        {
        	std::string term = doc_split[i];

        	/* Add term to vector list, if not done so already */
        	std::vector<std::string>::iterator term_it = std::find(term_vec.begin(), term_vec.end(), term);
        	if(term_it == term_vec.end())
            	term_vec.push_back(term);

        	/* Add the document to the list of documents for this term, if not done so already */
        	std::vector<int> doc_list = term_doc_map[term];
        	std::vector<int>::iterator doc_it = std::find(doc_list.begin(), doc_list.end(), lineno);
        	if(doc_it == doc_list.end())
            	doc_list.push_back(lineno);

        }

		lineno++;
	}

	/* Go through each term and populate the term_index_vec and doc_term_vec */
	for(int idx = 0; idx < term_vec.size(); idx++)
	{
		/* t is the term itself, idx is its index (in term_index_vec as well) */
		std::string t = term_vec[idx];

		/* d is the list of docs associated with t */
		std::vector<int> d = term_doc_map[t];

		/* The starting index for the list of docs (related to t) is the size of the doc_term_vec before we insert the new docs */
		term_index_vec.push_back(doc_term_vec.size());

		/* Insert the related documents in the doc_term_vec */
		doc_term_vec.insert(doc_term_vec.end(), d.begin(), d.end());

	}

	/* Convert the vectors to arrays for GPU processing */
	int *term_index_arr = vecToArr(term_index_vec);
	int *doc_term_arr = vecToArr(doc_term_vec);

	/* Create a TxC matrix (i.e. # of Terms x # of Classes) which will hold the frequencies of each term */
	float *term_class_matrix = (float *)calloc( (term_vec.size()) * (classes_vec.size()), sizeof(float) );

	
	/* Testing stuff */
	// std::cout << "There are " << term_vec.size() << " terms." << std::endl;
	// std::cout << "There are " << lineno << " docs." << std::endl;
	// std::cout << "There are " << classes_vec.size() << " classes." << std::endl;


}	
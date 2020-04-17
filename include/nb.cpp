/* Definition of Naive Bayes helper functions */
#include "./nb.hpp"

/* Tokenize string */
std::vector<std::string> tokenize(std::string s)
{
	std::istringstream iss(s);
	std::vector<std::string> doc_split((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());

	return doc_split;
}

/* Calculates log prior probability for a category */
float log_prior_prob(int categ_freq, int num_docs)
{
	return log(categ_freq / num_docs);
}

/* Calculates the log probability for a term in the category */
float log_term_categ_prob(float categ_term_count, float categ_total_terms)
{
	float prob = (categ_term_count + 1) / (categ_total_terms); 
	float log_prob = log(prob);

	return log_prob;
}

/* Perform training on a set of training docs */
void train(std::string filename, int &num_training_docs, std::vector<std::string> &term_vec, std::vector<std::string> &classes_vec, std::map<std::string,std::map<std::string,int> > &term_class_freq_map, std::map<std::string,std::vector<int> > &class_info_map)
{
	/* Loop through each document */
	std::ifstream file(filename);
	std::string line;

	while (std::getline(file, line)) 
	{
		/* 
			Tokenize the each line 
			doc_split[0] = doc_class
			doc_split[1 -> end] = terms in doc
		*/
		std::vector<std::string> doc_split = tokenize(line);
		std::string d_class = doc_split[0];

		/* Append class to classes_vec and create an association in the class_info_map, only if it has not been seen before */
		std::vector<std::string>::iterator class_it = std::find(classes_vec.begin(), classes_vec.end(), d_class);
        if(class_it == classes_vec.end())
        {
            classes_vec.push_back(d_class);

            /* Create a new class_info_vector for the class */
            std::vector<int> class_info_vec;
            class_info_vec.push_back(0); 		/* Number of documents */
            class_info_vec.push_back(0);		/* Number of terms */

            /* Make the association of the class to the class_info_vec */
            class_info_map[d_class] = class_info_vec;
        }

        class_info_map[d_class][0] += 1;

        /* Loop through each term in the document */
        for(int i = 1; i < doc_split.size(); i++)
        {
        	std::string term = doc_split[i];

        	/* Add term to vector list, if not done so already */
        	std::vector<std::string>::iterator term_it = std::find(term_vec.begin(), term_vec.end(), term);
        	if(term_it == term_vec.end())
            	term_vec.push_back(term);

        	/* If this class has not been associated this term, make the association */ 
        	if ( term_class_freq_map.count(term) == 0) 
  				term_class_freq_map[term][d_class] = 1;
			/* Otherwise, increase the number of times the term has appeared in the class */
			else 
  				term_class_freq_map[term][d_class] += 1;


  			/* Increase the number of terms in the class */
        	class_info_map[d_class][1] += 1; 

        }

        num_training_docs++;
	}

}


/* Do the learning for each (term,class) pair */
float * learn(std::vector<std::string> &term_vec, std::vector<std::string> &classes_vec, std::map<std::string,std::map<std::string,int> > &term_class_freq_map, std::map<std::string,std::vector<int> > &class_info_map)
{	
	/* Create the prob matrix */
	float *prob_matrix = (float *)calloc( (term_vec.size()) * (classes_vec.size()), sizeof(float) );

	if(prob_matrix == NULL)
	{
		std::cerr << "Error allocating prob_matrix" << std::endl;
		exit(-1);
	}

	/* Populate the matrix with the term and class info */
	for(int t_idx; t_idx < term_vec.size(); t_idx++)
	{
		std::string curr_term = term_vec[t_idx];

		for(int c_idx; c_idx < classes_vec.size(); c_idx++)
		{
			std::string curr_class = classes_vec[c_idx];
			float curr_freq = (float)term_class_freq_map[curr_term][curr_class];
			float curr_total_terms = (float)class_info_map[curr_class][1];

			float log_prob = log_term_categ_prob(curr_freq, curr_total_terms);

			*(prob_matrix + (t_idx * classes_vec.size()) + c_idx) = log_prob;
		}
	}

	return prob_matrix;
}


/* Perform the classification of test documents */
std::vector<int> test(std::string filename, int &num_training_docs, float *prob_matrix, std::vector<std::string> &term_vec, std::vector<std::string> &classes_vec, std::map<std::string,std::vector<int> > &class_info_map)
{
	/* Vector to hold the results */
	std::vector<int> results;

	/* Loop through each document */
	std::ifstream file(filename);
	std::string line;

	while (std::getline(file, line)) 
	{
		/* 
			Tokenize the each line 
			doc_split[0] = doc_class
			doc_split[1 -> end] = terms in doc

			We want to ignore the doc_class (otherwise this wouldn't be much of a project) 
		*/
		std::vector<std::string> doc_terms = tokenize(line);

		/* Create a map that counts how many times we have seen a specific term */
		std::map<std::string, int> term_count;

		/* Loop through each term in the document to get term counts */
        for(int i = 1; i < doc_terms.size(); i++)
        {
        	std::string term = doc_terms[i];

        	if(term_count.count(term) == 0)
        		term_count[term] = 1;
        	else
        		term_count[term] += 1;
        }

        /* Create vector to hold the total log probs of the class for each document */
        std::vector<float> all_class_log_probs;

        /* Loop through each category and compute the logprobs */
        for(int i = 0; i < classes_vec.size(); i++)
        {
        	/* Get priors */
        	int categ_freq = class_info_map[classes_vec[i]][0];
        	float log_prior = log_prior_prob(categ_freq, num_training_docs);

        	/* For each term, multiply the term count by the logprob in the prob matrix */
        	float log_class_prob = 0;
        	for(std::map<std::string, int>::iterator it = term_count.begin(); it != term_count.end(); ++it)
			{
				std::string term = it->first;
				int count = it->second;

				/* Check if term appeared in training docs and if so, multiply the prob by the num of times it appeared in this doc */
        		std::vector<std::string>::iterator term_it = std::find(term_vec.begin(), term_vec.end(), term);
        		if(term_it == term_vec.end())
            		log_class_prob += 0;
            	else
            	{
            		int t_idx = term_it - term_vec.begin();
            		int c_idx = i;
            		float log_prob = *(prob_matrix + (t_idx * classes_vec.size()) + c_idx);

            		log_class_prob += (log_prob * count); 
            	}

			}
        	
        	/* 
				Add the total prob to the list.
        		Note: Since this happens in each loop, the index in the below list should match the index of classes_vec 
        	*/
			all_class_log_probs.push_back(log_prior + log_class_prob);
        }


        int max_idx = std::distance(all_class_log_probs.begin(), std::max_element(all_class_log_probs.begin(), all_class_log_probs.end()));

        results.push_back(max_idx);
	}

	return results;

}
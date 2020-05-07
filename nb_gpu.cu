#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <cuda_profiler_api.h>
/*
 * Fills in the matrix term_class_matrix based on the frequency of terms. The term_index_arr
 * holds the indices for the doc_term_arr where each term starts. Increment frequency of
 * term_class_matrix for the class and term by looping through all docs with that term.
 * The doc_class array is used to hold the class of each doc
 */
__global__ void calcFreq(int *term_index_arr, int *doc_term_arr, int *doc_class, float *term_class_matrix,
							int num_terms, int doc_term_len, int classes) {
	unsigned int i = blockIdx.x * gridDim.y * gridDim.z *
                      blockDim.x + blockIdx.y * gridDim.z *
                      blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
	int start = term_index_arr[i];
	int end = term_index_arr[i];
	if(i < num_terms - 1) {
		end = term_index_arr[i+1];
	} else if (i == num_terms - 1){
		end = doc_term_len - 1;
	} else {
		return ;
	}

	for (int x = start; x < end; x++) {
		term_class_matrix[classes * i + doc_class[doc_term_arr[x] - 1]] += 1.0;
	}
}

/*
 * Calculates total number of terms per class and places into an array. Parallelized
 * based on class
 */
__global__ void calcTotalTermsPerClass(float * term_class_matrix, int * terms_per_class, int num_terms, int classes) {
	unsigned int i = blockIdx.x * gridDim.y * gridDim.z *
                      blockDim.x + blockIdx.y * gridDim.z *
                      blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
	if (i < classes) {
		int sum = 0;
		for (int x = 0; x < num_terms; x++) {
			sum += term_class_matrix[classes * x + i];
		}
		terms_per_class[i] = sum;
	}
}


/*
 * Goes through each term and divides the term frequency in the class by the total
 * terms in that class. Parallelized based on terms
 */
__global__ void learn(float * term_class_matrix, int num_docs, int classes, int * terms_per_class, int num_terms) {
	unsigned int i = blockIdx.x * gridDim.y * gridDim.z *
                      blockDim.x + blockIdx.y * gridDim.z *
                      blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
	if (i < num_terms) {
		for (int x = 0; x < classes; x++) {
			term_class_matrix[classes * i + x] /= terms_per_class[x];
		}
	}
}

__global__ void test(float *term_class_matrix, float * doc_prob, int * doc_index, int * terms_in_doc, int classes, int num_docs, int total_len_terms, int *predictions) {
	unsigned int i = blockIdx.x * gridDim.y * gridDim.z *
                      blockDim.x + blockIdx.y * gridDim.z *
                      blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
	int start_term = doc_index[i];
	int end_term = doc_index[i];
	if(i < num_docs - 1) {
		end_term = doc_index[i+1];
	} else if (i == num_docs - 1) {
		end_term = total_len_terms - 1;
	} else {
		return ;
	}
	for (int x = start_term; x < end_term; x++) {
		for (int y = 0; y < classes; y++) {
			doc_prob[classes * i + y] += log(term_class_matrix[classes * x + y]);
		}
	}
	int max_index = 0;
	float max = -3.40282e+038;
	for (int y = 0; y < classes; y++) {
		if (doc_prob[classes * i + y] > max) {
			max_index = y;
		}
	}
	predictions[i] = max_index;

}

void errorCheck(cudaError_t err) {
	if (err) {
		fprintf(stderr, "CUDA error: %d\n", err);
		exit(err);
	}
}

static cudaError_t numBlocksThreads(unsigned int N, dim3 *numBlocks, dim3 *threadsPerBlock) {
    unsigned int BLOCKSIZE = 128;
    int Nx, Ny, Nz;
    int device;
    cudaError_t err;
    if(N < BLOCKSIZE) {
        numBlocks->x = 1;
        numBlocks->y = 1;
        numBlocks->z = 1;
        threadsPerBlock->x = N;
        threadsPerBlock->y = 1;
        threadsPerBlock->z = 1;
        return cudaSuccess;
    }
    threadsPerBlock->x = BLOCKSIZE;
    threadsPerBlock->y = 1;
    threadsPerBlock->z = 1;
    err = cudaGetDevice(&device);
    if(err)
      return err;
    err = cudaDeviceGetAttribute(&Nx, cudaDevAttrMaxBlockDimX, device);
    if(err)
      return err;
    err = cudaDeviceGetAttribute(&Ny, cudaDevAttrMaxBlockDimY, device);
    if(err)
      return err;
    err = cudaDeviceGetAttribute(&Nz, cudaDevAttrMaxBlockDimZ, device);
    if(err)
      return err;
    unsigned int n = (N-1) / BLOCKSIZE + 1;
    unsigned int x = (n-1) / (Ny*Nz) + 1;
    unsigned int y = (n-1) / (x*Nz) + 1;
    unsigned int z = (n-1) / (x*y) + 1;
    if(x > Nx || y > Ny || z > Nz) {
        return cudaErrorInvalidConfiguration;
    }
    numBlocks->x = x;
    numBlocks->y = y;
    numBlocks->z = z;

    return cudaSuccess;
}

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

	std::vector<int> doc_class;

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

		doc_class.push_back(find(classes_vec.begin(), classes_vec.end(), doc_split[0]) - classes_vec.begin());

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
        	if(doc_it == doc_list.end()) {
				doc_list.push_back(lineno);
				term_doc_map[term] = doc_list;
			}
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

	std::ifstream test_file(argv[2]);
	/*
		Vector of Test documents.
		Each index represents a test document.
		The value at that index represents the index in test_term_doc_vec that holds list of terms for that documents
		Note: Will be converted to array later (to be used in kernel function)
	*/
	std::vector<int> test_doc_index_vec;

	/*
		Vector of valid test document terms.
		Each value represents the term_number that is valid and appears in the document
		Note: Will be converted to array later (to be used in kernel function)
	*/
	std::vector<int> test_term_doc_vec;

	while (std::getline(test_file, line)) {
		std::istringstream iss(line);
		std::vector<std::string> doc_split((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());

		std::vector<int> test_doc_terms;
		for(int i = 0; i < doc_split.size(); i++) {
			std::string term = doc_split[i];
			std::vector<std::string>::iterator term_it = std::find(term_vec.begin(), term_vec.end(), term);
			if (term_it != term_vec.end()) {
				test_doc_terms.push_back(term_it - term_vec.begin());
			} else {
				continue;
			}
		}

		test_doc_index_vec.push_back(test_term_doc_vec.size());
		test_term_doc_vec.insert(test_term_doc_vec.end(), test_doc_terms.begin(), test_doc_terms.end());
	}

	/* Convert the vectors to arrays for GPU processing */
	int *term_index_arr = vecToArr(term_index_vec);
	int *doc_term_arr = vecToArr(doc_term_vec);
	int *doc_class_arr = vecToArr(doc_class);

	/* Create a TxC matrix (i.e. # of Terms x # of Classes) which will hold the frequencies of each term */
	float *term_class_matrix = (float *)calloc( (term_vec.size()) * (classes_vec.size()), sizeof(float) );

	/* Create a C length array holding the total terms in each class*/
	int *total_terms_class_arr = (int *)calloc( classes_vec.size(), sizeof(int));

	size_t nSpatial;
	size_t mSpatial;
	dim3 spatialThreadsPerBlock, spatialBlocks;

	float *d_term_class;
	int *d_term_index;
	int *d_doc_term;
	int *d_doc_class;
	int *d_total_terms_class;

	cudaDeviceReset();
    cudaProfilerStart();


	nSpatial = doc_term_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(int);
	errorCheck(cudaMalloc(&d_doc_term, mSpatial));
	errorCheck(cudaMemcpy(d_doc_term, doc_term_arr, nSpatial*sizeof(int), cudaMemcpyHostToDevice));

	nSpatial = doc_class.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(int);
	errorCheck(cudaMalloc(&d_doc_class, mSpatial));
	errorCheck(cudaMemcpy(d_doc_class, doc_class_arr, nSpatial*sizeof(int), cudaMemcpyHostToDevice));

	nSpatial = term_vec.size() * classes_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(float);
	errorCheck(cudaMalloc(&d_term_class, mSpatial));
	errorCheck(cudaMemcpy(d_term_class, term_class_matrix, nSpatial*sizeof(float), cudaMemcpyHostToDevice));

	nSpatial = term_index_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(int);
	errorCheck(cudaMalloc(&d_term_index, mSpatial));
	errorCheck(cudaMemcpy(d_term_index, term_index_arr, nSpatial*sizeof(int), cudaMemcpyHostToDevice));

	// Learn
	calcFreq<<<spatialBlocks, spatialThreadsPerBlock>>>(d_term_index, d_doc_term, d_doc_class, d_term_class, term_vec.size(), doc_term_vec.size(), classes_vec.size());

	nSpatial = classes_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(int);
	errorCheck(cudaMalloc(&d_total_terms_class, mSpatial));
	errorCheck(cudaMemcpy(d_total_terms_class, total_terms_class_arr, nSpatial*sizeof(int), cudaMemcpyHostToDevice));

	calcTotalTermsPerClass<<<spatialBlocks, spatialThreadsPerBlock>>>(d_term_class, d_total_terms_class, term_vec.size(), classes_vec.size());

	nSpatial = term_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	learn<<<spatialBlocks, spatialThreadsPerBlock>>>(d_term_class, doc_class.size(), classes_vec.size(), d_total_terms_class, term_vec.size());

	// Test
	int *test_doc_index_arr = vecToArr(test_doc_index_vec);
	int *test_term_doc_arr = vecToArr(test_term_doc_vec);

	int *predictions = (int *) calloc(test_doc_index_vec.size(), sizeof(int));

	float *test_doc_prob = (float *)calloc( (test_doc_index_vec.size()) * (classes_vec.size()), sizeof(float) );
	float *d_test_doc_prob;

	int *d_test_doc_index;
	int *d_test_term_doc;
	int *d_predictions;

	nSpatial = test_doc_index_vec.size() * classes_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(float);
	errorCheck(cudaMalloc(&d_test_doc_prob, mSpatial));
	errorCheck(cudaMemcpy(d_test_doc_prob, test_doc_prob, nSpatial*sizeof(float), cudaMemcpyHostToDevice));

	nSpatial = test_term_doc_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(int);
	errorCheck(cudaMalloc(&d_test_term_doc, mSpatial));
	errorCheck(cudaMemcpy(d_test_term_doc, test_term_doc_arr, nSpatial*sizeof(int), cudaMemcpyHostToDevice));

	nSpatial = test_doc_index_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(int);
	errorCheck(cudaMalloc(&d_test_doc_index, mSpatial));
	errorCheck(cudaMemcpy(d_test_doc_index, test_doc_index_arr, nSpatial*sizeof(int), cudaMemcpyHostToDevice));

	errorCheck(cudaMalloc(&d_predictions, mSpatial));
	errorCheck(cudaMemcpy(d_predictions, predictions, nSpatial*sizeof(int), cudaMemcpyHostToDevice));

	test<<<spatialBlocks, spatialThreadsPerBlock>>>(d_term_class, d_test_doc_prob, d_test_doc_index, d_test_term_doc, classes_vec.size(), test_doc_index_vec.size(), test_term_doc_vec.size(), d_predictions);

	errorCheck(cudaMemcpy(predictions, d_predictions, nSpatial*sizeof(int), cudaMemcpyDeviceToHost));
	std::cerr << "Size of predictions: " << sizeof(predictions)/sizeof(int) << std::endl;
	std::cerr << "Size of tests: " << test_doc_index_vec.size() << std::endl;
	std::ofstream results("./results.txt");
	if(results.is_open()) {
		for (int i = 0; i < test_doc_index_vec.size(); i++) {
			results << classes_vec[predictions[i]] << '\n';
		}
	}

	/* Testing stuff */
	// std::cout << "There are " << term_vec.size() << " terms." << std::endl;
	// std::cout << "There are " << lineno << " docs." << std::endl;
	// std::cout << "There are " << classes_vec.size() << " classes." << std::endl;


}

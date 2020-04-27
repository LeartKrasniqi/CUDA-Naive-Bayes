#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <iterator>

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
	int sum = 0;
	for (int x = 0; x < num_terms; x++) {
		sum += term_class_matrix[classes * x + i];
	}
	terms_per_class[i] = sum;
}


/*
 * Goes through each term and divides the term frequency in the class by the total
 * terms in that class. Parallelized based on terms
 */
__global__ void learn(float * term_class_matrix, int num_docs, int classes, int * terms_per_class) {
	unsigned int i = blockIdx.x * gridDim.y * gridDim.z *
                      blockDim.x + blockIdx.y * gridDim.z *
                      blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
	for (int x = 0; x < classes; x++) {
		term_class_matrix[classes * i + x] /= terms_per_class[x];
	}
}

__global__ void test(float *term_class_matrix, float * doc_prob, int * term_index_arr, int * terms_in_doc, int classes, int num_docs, int total_len_terms) {
	unsigned int i = blockIdx.x * gridDim.y * gridDim.z *
                      blockDim.x + blockIdx.y * gridDim.z *
                      blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
	int start_term = term_index_arr[i];
	int end_term = term_index_arr[i];
	if(i < num_docs) {
		end_term = term_index_arr[i+1];
	} else {
		end_term = total_len_terms - 1;
	}
	for (int x = start_term; x < end_term; x++) {
		for (int y = 0; y < classes; y++) {
			doc_prob[classes * i + y] += term_class_matrix[classes * x + y];
		}
	}

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

	nSpatial = term_index_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(int);
	errorCheck(cudaMalloc(&d_term_index, mSpatial));
	errorCheck(cudaMemcpy(d_term_index, term_index_arr, nSpatial*sizeof(int), cudaMemcpyHostToDevice));

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



	// Learn
	calcFreq<<<spatialBlocks, spatialThreadsPerBlock>>>(d_term_index, d_doc_term, d_doc_class, d_term_class, term_vec.size(), term_vec.size(), classes_vec.size());

	nSpatial = classes_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	mSpatial = spatialBlocks.x * spatialBlocks.y * spatialBlocks.z * spatialThreadsPerBlock.x * sizeof(int);
	errorCheck(cudaMalloc(&d_total_terms_class, mSpatial));
	errorCheck(cudaMemcpy(d_total_terms_class, total_terms_class_arr, nSpatial*sizeof(int), cudaMemcpyHostToDevice));

	calcTotalTermsPerClass<<<spatialBlocks, spatialThreadsPerBlock>>>(d_term_class, d_total_terms_class, term_vec.size(), classes_vec.size());

	nSpatial = term_vec.size();
	errorCheck(numBlocksThreads(nSpatial, &spatialBlocks, &spatialThreadsPerBlock));
	learn<<<spatialBlocks, spatialThreadsPerBlock>>>(d_term_class, doc_class.size(), classes_vec.size(), d_total_terms_class);






	/* Testing stuff */
	// std::cout << "There are " << term_vec.size() << " terms." << std::endl;
	// std::cout << "There are " << lineno << " docs." << std::endl;
	// std::cout << "There are " << classes_vec.size() << " classes." << std::endl;


}

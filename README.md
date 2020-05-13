# CUDA-Naive-Bayes

## Requirements
```
cuda 10 toolkit
gcc
openMP
git
```

We suggest having this repo cloned on both the cross-compile machine and the target machine.  That way you have access to the relevant datasets and metric scripts.

**Note:** If you do not have `openMP`, you can remove the flag in the `CXXFLAGS` variable in the Makefile.

## GPU Usage
1. Run on the cross-compile machine:
```bash
cd CUDA-Naive-Bayes
make nb_gpu
```
2. Results in `nb_gpu` executable.  Move this to the machine with the GPU (e.g. Jetson Nano).

3. On the Jetson Nano:
```bash
cd /path/to/executable/
./nb_gpu [train_file] [test_file] [output_file]
```

## CPU Usage
```bash
cd CUDA-Naive-Bayes
make nb_cpu.out
./nb_cpu.out [train_file] [test_file] [output_file]
```

## Accuracy Metrics
```bash
python3 CUDA-Naive-Bayes/metrics/calcF1.py [classifier_output] [true_output]
```

## Example Using Provided Datasets
Assuming you have already made the relevant executable:
```bash
cd CUDA-Naive-Bayes

# Run the classifier
/path/to/executable ./docs/train/[train_file] ./docs/test/[test_file] output.txt

# Calculate accuracy metrics
python3 ./metrics/calcF1.py output.txt ./docs/labels/[true_output]
```

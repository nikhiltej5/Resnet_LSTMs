import subprocess
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--model_file', type=str, default='model.py', help='Path to model file')
    parser.add_argument('--test_data_file', type=str, default='data/test', help='Path to test data')
    parser.add_argument('--normalization', type=str, default='BN', help='Normalization to use')
    parser.add_argument('--n', type=str, default='2', help='Number of layers in each block')
    parser.add_argument('--output_file', type=str, default='output.txt', help='Path to output file')
    args = parser.parse_args()
    if (args.normalization == 'inbuilt'):
        subprocess.run(['python', 'ass1.py', '--model_file', args.model_file, '--test_data_file', args.test_data_file, '--n', args.n, '--output_file', args.output_file])
    elif (args.normalization == 'bn'):
        subprocess.run(['python', 'ass1BN.py', '--model_file', args.model_file, '--test_data_file', args.test_data_file, '--n', args.n, '--output_file', args.output_file])
    elif (args.normalization == 'ln'):
        subprocess.run(['python', 'ass1LN.py', '--model_file', args.model_file, '--test_data_file', args.test_data_file, '--n', args.n, '--output_file', args.output_file])
    elif (args.normalization == 'nn'):
        subprocess.run(['python', 'ass1NN.py', '--model_file', args.model_file, '--test_data_file', args.test_data_file, '--n', args.n, '--output_file', args.output_file])
    elif (args.normalization == 'in'):
        subprocess.run(['python', 'ass1IN.py', '--model_file', args.model_file, '--test_data_file', args.test_data_file, '--n', args.n, '--output_file', args.output_file])
    elif (args.normalization == 'bin'):
        subprocess.run(['python', 'ass1BIN.py', '--model_file', args.model_file, '--test_data_file', args.test_data_file, '--n', args.n, '--output_file', args.output_file])
    elif (args.normalization == 'gn'):
        subprocess.run(['python', 'ass1GN.py', '--model_file', args.model_file, '--test_data_file', args.test_data_file, '--n', args.n, '--output_file', args.output_file])
    else:
        print("Invalid normalization")
        exit(1)

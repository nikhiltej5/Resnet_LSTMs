# Inference Script

This script (`infer.py`) is designed to run inference on trained models. It provides functionality to test models with different configurations.

## Usage

### For models with 6n + 2 layers:

```bash
python3 infer.py --model_file <path_to_model> --normalization [bn | in | bin | ln | gn | nn | inbuilt] --n [ 2 ] --test_data_file <path_to_test_data> --output_file <output_file_path>

```
The `output_file` will contain predicted class ids for each image in the input directory.

### For Seq2Seq models

```bash
python3 infer.py --model_file <path_to_model> --beam_size [1 | 10 | 20] --model_type [lstm_lstm | lstm_lstm_attn | bert_lstm_attn_frozen | bert_lstm_attn_tuned] --test_data_file <path_to_test_data>

```
The predictions will be written back to the same JSON file with an extra field named `predicted`.


Feel free to adjust or expand upon this as needed.







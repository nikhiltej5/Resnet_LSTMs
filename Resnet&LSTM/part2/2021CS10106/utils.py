from torchtext.data.utils import get_tokenizer
import pickle
import json

def tokenize_linear_formula(linear_formula):
    # Split the linear formula string by the pipe character "|"
    operations = linear_formula.split('|')
    
    # Initialize a list to store tokenized operations
    tokenized_operations = []
    
    # Iterate through each operation
    for operation in operations:
        if (len(operation) > 0):
            tokens = operation.split('(')
            if (len(tokens) > 1):
                for token in tokens:
                    if (len(token) > 0):
                        tok = token.split(')')
                        if (len(tok) > 1):
                            for T in tok:
                                if (len(T) > 0):
                                    t = T.split(',')
                                    if (len(t) > 1):
                                        for temp in t:
                                            if (len(temp) > 0):
                                                tokenized_operations.append(temp)
                                                tokenized_operations.append(',')
                                            else:
                                                tokenized_operations.append(',')
                                        tokenized_operations.pop()
                                    else:
                                        tokenized_operations += t
                                    tokenized_operations.append(')')
                                else:
                                    tokenized_operations.append(')')
                            tokenized_operations.pop()
                        else:
                            tokenized_operations += tok
                        tokenized_operations.append('(')
                    else:
                        tokenized_operations.append(')')
                tokenized_operations.pop()
            else:
                tokenized_operations += tokens
            tokenized_operations.append('|')
    tokenized_operations.pop()
    return tokenized_operations

# Function to tokenize text data
def tokenize_text(text):
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(text)
    return tokens

# Function to convert text data into numerical sequences
def out_text_to_numerical(Text):
    with open("/processing/decoderword2idx.pickle", "rb") as file:
        word2idx2 = pickle.load(file)
    tokens = tokenize_linear_formula(Text)
    output = [word2idx2.get(x) for x in tokens]
    return output

def text_to_numerical(text, vocab):
    tokens = tokenize_text(text)
    numerical_sequence = [vocab.stoi[token] for token in tokens if token in vocab.stoi]
    return numerical_sequence

def numerical_to_text(sequence, glove):
    text = [glove.itos[num] for num in sequence]
    return " ".join(text)

def out_numerical_to_text(sequence):
    with open("/processing/decoderidx2word.pickle", "rb") as file:
        idx2word2 = pickle.load(file)
    text = [idx2word2.get(x) for x in sequence]
    return "".join(text)

# Function to preprocess data
def preprocess_data(data_path, glove):
    with open(data_path, "r") as f:
        data = json.load(f)
    
    processed_data = []
    for example in data:
        problem_text = example["Problem"]
        linear_formula = example["linear_formula"]
        answer = example["answer"]
        
        # Convert text data into numerical sequences
        problem_embedding = text_to_numerical(problem_text, glove)
        linear_formula_embedding = out_text_to_numerical(linear_formula)

        # Append processed data
        processed_data.append({"Problem": problem_embedding, "linear_formula": linear_formula_embedding, "answer": answer})
    
    return processed_data
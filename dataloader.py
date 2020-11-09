import tensorflow as tf

def dataset_for_generator(data_file, batch_size):
    token_stream = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            
            # 今回は文字列の長さは指定しないので
#             if len(parse_line) == 20:
            token_stream.append(parse_line)
    return tf.data.Dataset.from_tensor_slices(token_stream).shuffle(len(token_stream)).batch(batch_size)

def dataset_for_discriminator(positive_file, negative_file, batch_size):
    examples = []
    labels = []
    with open(positive_file) as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
#             if len(parse_line) == 20:
            examples.append(parse_line)
            labels.append([0, 1])
        
    with open(negative_file) as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
#             if len(parse_line) == 20:
            examples.append(parse_line)
            labels.append([1, 0])
    return tf.data.Dataset.from_tensor_slices((examples, labels)).shuffle(len(examples)).batch(batch_size)

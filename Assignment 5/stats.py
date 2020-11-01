def main():
    X_train, Y_train = [], []
    with open('training_data.tsv', 'r') as f:
        for line in f:
            split = line.strip().split('\t')
            # Get feature representation
            #embedding_1 = get_embedding(split[0], embeddings)
            #embedding_2 = get_embedding(split[1], embeddings)
            #X_train.append(embedding_1 + embedding_2)
            # Get label
            label = split[2]
            Y_train.append(label)


if __name__ == '__main__':
    main()

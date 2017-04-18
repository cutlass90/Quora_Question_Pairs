from siameseRNN import SiameseRNN
import tools


siamese_RNN = SiameseRNN(embedding_size=300, n_hidden_RNN=128,
    do_train=True)
"""
# Training
data_provider = tools.DataProvider(path_to_csv='dataset/train.csv',
    path_to_w2v='embeddings/GoogleNews-vectors-negative300.bin',
    # path_to_w2v='~/GoogleNews-vectors-negative300.bin',    
    test_size=0.15)

siamese_RNN.train_(data_loader=data_provider, keep_prob=1, weight_decay=0.05,
    learn_rate_start=0.005, learn_rate_end=0.0003, batch_size=64, n_iter=100000,
    save_model_every_n_iter=5000, path_to_model='models/siamese')


#Evaluating COST
siamese_RNN.eval_cost(data_loader=data_provider, batch_size=512,
    path_to_model='models/siamese')
"""


#Prediction
data_provider = tools.DataProvider(path_to_csv='dataset/test.csv',
    path_to_w2v='embeddings/GoogleNews-vectors-negative300.bin',
    # path_to_w2v='~/GoogleNews-vectors-negative300.bin',    
    test_size=0)

siamese_RNN.predict(batch_size=512, data_loader=data_provider, path_to_save='submission.csv',
    path_to_model='models/siamese')

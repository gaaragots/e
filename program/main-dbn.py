import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from deps.util import log_to_file, tokeniser, save_model, load_model
import os.path
from sklearn.model_selection import train_test_split
from dbn.models import UnsupervisedDBN

log_file_name = 'kfold - add hidden layers.log'

@log_to_file(log_file_name)
def main(tfidfModel=None, tfidfMatrix=None, dbn_model=None, kmeans_model=None):
    # 0. read data and splite it to 80% for training and 20% for testing
    items_info  = pd.read_csv('input/items.csv', sep=';', encoding='ISO-8859-1')
    u_base1     = pd.read_csv('input/u1.base', sep='\t', header=None)
    train       = pd.DataFrame(u_base1[1].drop_duplicates())
    u_test1     = pd.read_csv('input/u1.test', sep='\t', header=None)
    test        = pd.DataFrame(u_test1[1].drop_duplicates())

    train_desc = [items_info[items_info['movie id'] == df[1]]['movie desription'] for i, df in train.iterrows()]
    test_desc  = [items_info[items_info['movie id'] == df[1]]['movie desription'] for i, df in test.iterrows()]


    # 1. train tf-idf model and save it under model/tf-idf-model.pickle with the result
    if not tfidfModel:
        print('traning tf-idf model ...')
        tfidf_model = TfidfVectorizer(norm='l2',min_df=0, use_idf=True,max_features=5000, smooth_idf=False, sublinear_tf=True, tokenizer=tokeniser)
        print('- Saving tf-idf model ...')
        save_model('model/tfidf_model.pickle', tfidf_model)
    else:
        print('# Loading tf-idf model ...')
        tfidf_model = load_model(tfidfModel)


    if not tfidfMatrix:
        item_feature_matrix = tfidf_model.transform(train_desc)
        # 1.2. saving tf-idf matrix result
        print('- Saving tf-idf matrix result ...')
        save_model('result/item_feature_matrix.pickle', item_feature_matrix)
    else:
        print('# Loading tf-idf matrix result ...')
        item_feature_matrix = load_model(tfidfMatrix)
    

    if not dbn_model:
        dbn = UnsupervisedDBN(hidden_layers_structure=[5000, 400],
                              batch_size=10,
                              learning_rate_rbm=0.06,
                              n_epochs_rbm=20,
                              activation_function='sigmoid')
        # 2.2. fit dbn model
        dbn.fit(item_feature_matrix.A)
        # 2.3. save dbn model
        print('saving DBN model ...')
        dbn.save('model/dbn-model.pkl')
    else:
        print('Loadin DBN model')
        dbn = UnsupervisedDBN.load(dbn_model)


    # 3. Clustering with k-mens and save model and results
    if not kmeans_model:
        kmeans = KMeans(n_clusters=5, random_state=0).fit(dbn.transform(item_feature_matrix.A))
        print('saving k-means model ...')
        save_model('model/kmeans-model.pkl', kmeans)
    else:
        print('loading k-means model ...')
        kmeans = load_model(kmeans_model)
    
    print("Done!")


if __name__ == '__main__':
    tfidfModel = 'model/tfidf_model.pickle'
    tfidfMatrix = 'result/item_feature_matrix.pickle'
    dbn_model = 'model/dbn-model.pkl'
    kmeans_model = 'model/kmeans-model.pkl'

    # Load already trained model
    main(tfidfModel, tfidfMatrix, dbn_model, kmeans_model)
    
    # Train the model from the begining
    #main()

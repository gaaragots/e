import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from deps.util import log_to_file, tokeniser, save_model, load_model
import os.path
from sklearn.model_selection import train_test_split
from dbn.models import UnsupervisedDBN

@log_to_file('example.log')
def main():
    # 0. read data and splite it to 80% for training and 20% for testing
    items = pd.read_csv('input/items.csv', sep=';', encoding='ISO-8859-1')
    
    print items.shape
    
    items_train, items_test = train_test_split(items, train_size=0.8, random_state=0)
    
    print items_train.shape, items_test.shape


    # 1. train tf-idf model and save it under model/tf-idf-model.pickle with the result
    if not os.path.isfile('model/tfidf_model.pickle'):
        print('traning tf-idf model ...')
        tfidf_model = TfidfVectorizer(norm='l2',min_df=0, use_idf=True,max_features=5000, smooth_idf=False, sublinear_tf=True, tokenizer=tokeniser)
        item_feature_matrix = tfidf_model.fit_transform(items_train['movie desription'].values.astype('U'))
        print('#1. dimension of the item-feature matrix', item_feature_matrix.shape)

        # 1.1 saving tf-idf model
        print('Saving tf-idf model ...')
        save_model('model/tfidf_model.pickle', tfidf_model)
    
    if not os.path.isfile('result/item_feature_matrix.pickle'):
        # 1.2. saving tf-idf matrix result
        print('Saving tf-idf matrix result ...')
        save_model('result/item_feature_matrix.pickle', item_feature_matrix)

    # 2. train dbn model and save the model into model/dbn.pickle
    # 2.1. load tf-idf result
    print('loading item feature matrix ...')
    item_feature_matrix = load_model('result/item_feature_matrix.pickle')
    
    if not os.path.isfile('model/dbn-model.pkl'):
        dbn = UnsupervisedDBN(hidden_layers_structure=[15171, 4000],
                              batch_size=10,
                              learning_rate_rbm=0.06,
                              n_epochs_rbm=20,
                              activation_function='sigmoid')
        # 2.2. fit dbn model
        dbn.fit(item_feature_matrix.A)
        # 2.3. save dbn model
        print('saving DBN model ...')
        dbn.save('model/dbn-model.pkl')
        
    print('Loadin DBN model')
    dbn = UnsupervisedDBN.load('model/dbn-model.pkl')

    # 3. Clustering with k-mens and save model and results
    if not os.path.isfile('model/kmeans-model.pkl'):
        kmeans = KMeans(n_clusters=5, random_state=0).fit(dbn.transform(item_feature_matrix.A))
        print('saving k-means model ...')
        save_model('model/kmeans-model.pkl', kmeans)
    else:
        kmeans = load_model('model/kmeans-model.pkl')
    
    print(kmeans.labels_)
    

if __name__ == '__main__':
    main()

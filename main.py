# collaborative system - predicts what you like based on other similar users have liked in past
# content based system -  predicts what you like based on what you have liked in past

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM 

# fetch data n format it
data = fetch_movielens(min_rating=4.0)

# training and testing data
# print(repr(data['train']))
# print(repr(data['test']))
print(data['item_labels'][:10])


# create model
# loss measures the difference between model's prediction and desired output 
# minimize it during training , so model gets more accurate in prediction
model = LightFM(loss='logistic')

# train model
model.fit(data['train'],epochs=30,num_threads=2)

def sample_recommendation(model,data,user_ids): 
    # no of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recom. for each user we input
    # get a list of positive for each i.e 5 rating ... 4 or below is negative
    for user_id in user_ids: 
        
        # movies they already like 
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts
        scores = model.predict(user_id,np.arange(n_items))

        # rank in order of most to least liked
        top_items =  data['item_labels'][np.argsort(-scores)]

        print("User  %s " % user_id)
        print("  Known positives : ")

        for x in known_positives[:3]:
            print( "        %s" % x)

        print("   Recommended : ")    

        for x in top_items[:3]:
            print( "    %s " % x)

sample_recommendation(model,data,[1,35,350])
import csv

train_set = []
test_set = []

with open('train_100k_withratings.csv', mode="r") as csv_train:
    reader = csv.reader(csv_train)

    for entry in reader:
        train_set.append(entry)

with open('test_100k_withoutratings.csv', mode="r") as csv_test:
    reader = csv.reader(csv_test)

    for entry in reader:
        test_set.append(entry)
        
user_item_ratings = {}
item_user_ratings = {}

# Iterate over the train_set to populate the user_item_ratings and item_user_ratings dictionaries
for entry in train_set:
    user_id = entry[0]
    item_id = entry[1]
    rating = entry[2]
    
    # Update user_item_ratings
    if user_id in user_item_ratings:
        user_item_ratings[user_id].append({item_id: rating})
    else:
        user_item_ratings[user_id] = [{item_id: rating}]
    
    # Update item_user_ratings
    if item_id in item_user_ratings:
        item_user_ratings[item_id][user_id] = rating
    else:
        item_user_ratings[item_id] = {user_id: rating}

user_item_list = [[user_id, items_ratings] for user_id, items_ratings in user_item_ratings.items()]
item_user_list = [[item_id, user_ratings] for item_id, user_ratings in item_user_ratings.items()]
#print(item_user_list)
        
# calculate every users average rating [user_id, average_rating]
user_average_rating = []
for user_id, items_ratings in user_item_list:
    itemSize = len(items_ratings)
    ratingSum = sum(float(list(item_rating.values())[0]) for item_rating in items_ratings)
    avg = ratingSum / itemSize
    
    user_average_rating.append([user_id, avg])
#print(user_average_rating)

# create a list of all items e.g [item_id]
items_list = []
for entry in train_set:
    item_id = entry[1]
    
    if not (item_id in items_list):
        items_list.append(item_id)
        
items_list = [int(item_id) for item_id in items_list]
items_list.sort(reverse=False)
#print(items_list)

items_matrix = []
# create item matrix to later add on the cosine similarity
#for itemA in items_list:
    #for itemB in items_list:
        #items_matrix.append((itemA, itemB))
for i in range(len(items_list)):
    for j in range(i+1, len(items_list)):  # Start from i+1 to avoid duplicates
        items_matrix.append((items_list[i], items_list[j]))
#print(items_matrix)

#print(item_user_list)

#calculate cosine similarity of all items e.g [[item_id1,item_id2], cos_similarity]
cos_similarity = []
for itemA, itemB in items_matrix:
    if itemA == itemB:
        cos_similarity.append(((itemA, itemB), 1.00))
    else:
        itemA_usersWithScores = {}
        itemB_usersWithScores = {}
        
        for item, userScores in item_user_list:
            if int(item) == itemA:
                itemA_usersWithScores.update(userScores)

            if int(item) == itemB:
                itemB_usersWithScores.update(userScores)
                
        # extract common user keys between itemA and itemB
        common_keys = set(itemA_usersWithScores.keys()) & set(itemB_usersWithScores.keys())
        # append common keys and scores 
        if not common_keys:
            common_data = {}
        else:
            common_data = {k: [itemA_usersWithScores[k], itemB_usersWithScores[k]] for k in common_keys}
            
        # sum for top part of equation
        numerator = 0
        sumBelowA = 0
        sumBelowB = 0
        for key, value in common_data.items():
            average_rating = None
            for user_id, rating in user_average_rating:
                if user_id == key:
                    average_rating = rating
                    break      
                
            rating_user_A = float(value[0])
            rating_user_B = float(value[1])          
            
            numerator = numerator + (rating_user_A - average_rating)*(rating_user_B - average_rating)
            sumBelowA += (rating_user_A - average_rating) ** 2
            sumBelowB += (rating_user_B - average_rating) ** 2
        
        
        denominator = (sumBelowA ** 0.5) * (sumBelowB ** 0.5)
        if denominator == 0:
            cos_similarity.append(((itemA, itemB), 0.0))  # Avoid division by zero
        else:
            sim = numerator / denominator
            cos_similarity.append(((itemA, itemB), sim))

        cos_similarity.append(((itemA, itemB), sim))
        #print("Cos Similarity: ", cos_similarity)
        
print(cos_similarity)

import csv
import os
import random

train_set_full = []
with open('train_100k_withratings.csv', mode="r") as csv_train:
    reader = csv.reader(csv_train)

    for entry in reader:
        train_set_full.append(entry)
        
# Shuffle the training set randomly
random.shuffle(train_set_full)

# Determine the size of the subset (20% of the original training set)
subset_size = int(0.2 * len(train_set_full))

# Use the first subset_size entries as the subset
train_set = train_set_full[:subset_size]
        
user_item_list = {}
item_user_list = {}
# Iterate over the train_set to populate the user_item_ratings and item_user_ratings dictionaries
for entry in train_set:
    user_id = entry[0]
    item_id = entry[1]
    rating = entry[2]
    
    # Update user_item_ratings
    if user_id in user_item_list:
        user_item_list[user_id].append({item_id: rating})
    else:
        user_item_list[user_id] = [{item_id: rating}]
    
    # Update item_user_ratings
    if item_id in item_user_list:
        item_user_list[item_id][user_id] = rating
    else:
        item_user_list[item_id] = {user_id: rating}
                
# calculate every users average rating [user_id, average_rating]
user_average_rating = []
for user_id, items_ratings in user_item_list.items():
    itemSize = len(items_ratings)
    ratingSum = sum(float(list(item_rating.values())[0]) for item_rating in items_ratings)
    avg = ratingSum / itemSize
    
    user_average_rating.append([user_id, avg])

# create a list of all items e.g [item_id]
items_list = []
for entry in train_set:
    item_id = entry[1]
    
    if not (item_id in items_list):
        items_list.append(item_id)
        
items_list = [int(item_id) for item_id in items_list]
items_list.sort(reverse=False)

items_matrix = []
for i in range(len(items_list)):
    for j in range(len(items_list)):
        # Add diagonal elements
        if i == j:
            items_matrix.append((items_list[i], items_list[j]))
        # Add non-duplicate pairs
        elif i < j:
            items_matrix.append((items_list[i], items_list[j]))

#calculate cosine similarity of all items e.g [[item_id1,item_id2], cos_similarity]
cos_similarity = []
# check if the cosine similarity file exists
if os.path.exists("cosine_similarity.txt"):
    # if the file exists, load the cosine similarity matrix from the file
    with open("cosine_similarity.txt", "r") as file:
        cos_similarity = [eval(line.strip()) for line in file]
        
    print("cosine_similarity.txt already exists, to see changes made delete the file and rerun the code.")
else:
    for itemA, itemB in items_matrix:
        sim = 0
        if itemA == itemB:
            cos_similarity.append([(itemA, itemB), 1.00])
        else:
            itemA_usersWithScores = item_user_list.get(str(itemA), {})
            itemB_usersWithScores = item_user_list.get(str(itemB), {})
        
    
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
            
                numerator += (rating_user_A - average_rating)*(rating_user_B - average_rating)
                sumBelowA += (rating_user_A - average_rating) ** 2
                sumBelowB += (rating_user_B - average_rating) ** 2
        
            denominator = (sumBelowA ** 0.5) * (sumBelowB ** 0.5)
            if denominator == 0:
                sim = 0  # Avoid division by zero
            else:
                sim = numerator / denominator
            
            cos_similarity.append([(itemA, itemB), sim])
                
    # Open a file for writing
    with open("cosine_similarity.txt", "w") as file:
        # Iterate through the cosine similarity list and write each entry to the file
        for i, entry in enumerate(cos_similarity):
            file.write(f"{entry}\n")  # Write each entry to a new line
        
            # Print progress message every 10000 entries
            if (i + 1) % 10000 == 0:
                print(f"{i + 1} entries written to file...")

    # Notify the user that the cosine similarity matrix has been logged to the file
    print("Cosine similarity matrix has been logged to cosine_similarity.txt")

def predict_rating(user_id, item_id):
    weighted_sum = 0
    total_similarity = 0
    rating = 0
    
    # Check if the user has rated any items
    if user_id in user_item_list:
        neighborhood = []  # List to store similar items in the neighborhood
        
        # Iterate over cosine similarity values
        for (itemA, itemB), similarity in cos_similarity:
            # Check if the current item is one of the two items being compared
            if itemA == int(item_id) or itemB == int(item_id):
                other_item_id = itemA if itemB == int(item_id) else itemB
                
                # Check if the user has rated the other item
                if str(other_item_id) in item_user_list and str(user_id) in item_user_list[str(other_item_id)]:
                    other_item_rating = float(item_user_list[str(other_item_id)][str(user_id)])
                    
                    # Accumulate weighted sum and total similarity
                    weighted_sum += similarity * other_item_rating
                    if similarity > 0:
                        total_similarity += similarity
                        
                    # Store similar items in the neighborhood
                    neighborhood.append((other_item_id, similarity))
        
        # Normalize total_similarity
        if total_similarity != 0:
            for i in range(len(neighborhood)):
                neighborhood[i] = (neighborhood[i][0], neighborhood[i][1] / total_similarity)
                
        # Sort neighborhood by similarity in descending order
        neighborhood.sort(key=lambda x: x[1], reverse=True)
        
        # Determine the final rating based on the neighborhood
        if len(neighborhood) > 0:
            for i in range(min(len(neighborhood), 20)):  # Consider up to 20 most similar items
                weighted_sum += float(item_user_list[str(neighborhood[i][0])][str(user_id)])
            
            rating = weighted_sum / min(len(neighborhood), 20)  # Average rating from the neighborhood
            
            rating = min(rating, 5.0)
    else:
        rating = 2.5  # Default rating if the user has not rated any items
    
    return rating

# Load test set
test_set = []
with open('test_100k_withoutratings.csv', mode="r") as csv_test:
    reader = csv.reader(csv_test)
    for entry in reader:
        test_set.append(entry)

# Predict ratings for each entry in the test set
predicted_ratings = []
for user_id, item_id, timestamp in test_set:
    predicted_rating = predict_rating(user_id, item_id)
    predicted_ratings.append([user_id, item_id, predicted_rating, timestamp])

# Save the updated test set with predicted ratings
with open("test_100k_withpredictedratings.csv", "w", newline='') as csv_predicted:
    writer = csv.writer(csv_predicted)
    writer.writerows(predicted_ratings)

print("Predicted ratings have been added to test_100k_withpredictedratings.csv")
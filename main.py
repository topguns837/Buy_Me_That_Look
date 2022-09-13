import streamlit as st
import os
import cv2
from PIL import Image
import pandas as pd
import model_definition
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from pose_estimation import PoseEstimation
from yolo import *


os.chdir(os.getcwd())

filename = "input.jpg"

#value = st.slider('val')  # this is a widget
#st.write(value, 'squared is', value * value)

st.write("Myntra Fashion Recommender")

def save_uploadedfile(uploadedfile):
    global filename
    #filename = uploadedfile.name

    with open(os.path.join("temp","input.jpg"),"wb") as f:
        f.write(uploadedfile.getbuffer())    
    

    return st.success("Successfuly uploaded file ")
     

image_file = st.file_uploader("Enter a full-shot image", type = ['png','jpeg','jpg'])

if image_file is not None :
    save_uploadedfile(image_file)

    ps = PoseEstimation()
    print(filename)

    image_path = os.path.join("temp", filename)
    print(image_path)    

    if ps.driver(image_path) :
        st.write("Full shot")
        driver('temp/input.jpg')
        yolo_output = Image.open('temp/yolo_output.jpg')
        st.image(yolo_output, caption='Input Image')

        data_json = pd.read_csv(os.path.join('Product', 'data_json_features.csv'))
        
        for file in os.listdir('user_products') :
            print("for loop")
            new_embedding = model_definition.extract_features(os.path.join('user_products', file), model_definition.model)
            new_embedding_arr = []

            for i in range(len(new_embedding)) :
                new_embedding_arr.append(new_embedding[i])

            new_embedding = np.array(new_embedding)

            scaler = StandardScaler()
            new_feature_scaled = scaler.fit_transform(new_embedding.reshape(-1,1))
            new_feature_scaled = np.array(new_feature_scaled).squeeze()

            old_embedding = np.load('Product/features_pca.pkl', allow_pickle = True)
            old_features = np.load('Product/features_np.pkl', allow_pickle = True)

            topwear_features = old_features[0:70]
            bottomwear_features = old_features[71:159]
            footwear_features = old_features[160 : 259]
            eyewear_features = old_features[260 : 358]
            handbag_features = old_features[359 : 458]

            base_index = 0

            neighbors, distances, indices = [], [], []
            
            if file == "Topwear.jpg" :
                neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(topwear_features)
                distances, indices = neighbors.kneighbors([new_feature_scaled])
                base_index = 0
                st.write("\n Topwear Recommendations \n")
            
            elif file == "Bottomwear.jpg" :
                neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(bottomwear_features)
                distances, indices = neighbors.kneighbors([new_feature_scaled])
                base_index = 71
                st.write("\n Bottomwear Recommendations \n")

            elif file == "Footwear.jpg" :
                neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(footwear_features)
                distances, indices = neighbors.kneighbors([new_feature_scaled])
                base_index = 160
                st.write("\n Footwear Recommendations \n")

            elif file == "Eyewear.jpg" :
                neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(eyewear_features)
                distances, indices = neighbors.kneighbors([new_feature_scaled])
                base_index = 260
                st.write("\n Eyewear Recommendations \n")           
                

            elif file == "Handbag.jpg" :
                eighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(handbag_features)
                distances, indices = neighbors.kneighbors([new_feature_scaled])
                base_index = 359
                st.write("\n Handbag Recommendations \n")

            else:
                pass



            indexes = indices
            print(indexes)

            root_dir = 'Product'

            filenames = []
            category = data_json.iloc[indices[0] + base_index,:]['category']

            names = []
            prices = []
            links = []

            for index in indexes[0] :
                img_path = os.path.join(root_dir, data_json.iloc[index + base_index,:]['category'], data_json.iloc[index + base_index,:]['name']) + '.jpg'
                filenames.append(img_path)
                links.append(data_json.iloc[index + base_index,:]['link'])
                names.append(data_json.iloc[index + base_index, :]['name'])
                prices.append(data_json.iloc[index + base_index, : ]['price'])

            #print(filenames)

            counter = 0

            try :

                for img_path in filenames :
                    #img = cv2.imread(img_path)
                    img = Image.open(img_path)
                    #img = img.resize((600, 400))
                    st.image(img, caption='test', width  = 200)
                    st.write(names[counter])
                    st.write("Price : " +  str(prices[counter]) + " \u20B9")
                    st.write(links[counter])
                    counter += 1

            except :
                pass








    else:
        st.error("Not  full shot")

   








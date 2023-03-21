import streamlit as st
import os
import glob
import cv2
from PIL import Image
import pandas as pd
import numpy as np
import model_definition
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

from pose_estimation import PoseEstimation
from yolo import driver

import warnings
warnings.filterwarnings("ignore")


def clear_folder(path) :
    [f.unlink() for f in Path(path).glob("*") if f.is_file()] 

def save_uploadedfile(uploadedfile):
    #global filename   

    with open(os.path.join("temp","input.jpg"),"wb") as f:
        f.write(uploadedfile.getbuffer())    
    

    return st.success("Successfuly uploaded file ")
    


os.chdir(os.getcwd())

st.header("Myntra Fashion Recommender")

sex = None

sex = st.radio(
"Enter your gender : ",
('Male', 'Female'))

if sex == "Male" :
    sex = "Men"
else :
    sex = "Women"



     

image_file = st.file_uploader("Enter a full-shot image", type = ['png','jpeg','jpg'])

if image_file is not None :
    save_uploadedfile(image_file)

    ps = PoseEstimation()
    

    image_path = os.path.join("temp", "input.jpg")
       

    if ps.driver(image_path) :
        
        driver('temp/input.jpg')
        yolo_output = Image.open('temp/yolo_output.jpg')
        st.image(yolo_output, caption='Input Image')

        data_json = pd.read_csv(os.path.join('Product', 'data_json_new_copy.csv'))
        


        if True:

        
            for file in os.listdir('user_products') :

                try :                                   
                    new_embedding = model_definition.extract_features(os.path.join('user_products', file), model_definition.model)

                

                    new_embedding_arr = []

                    for i in range(len(new_embedding)) :
                        new_embedding_arr.append(new_embedding[i])

                    new_embedding = np.array(new_embedding)               
                
                

                    scaler = StandardScaler()
                    new_feature_scaled = scaler.fit_transform(new_embedding.reshape(-1,1))
                    new_feature_scaled = np.array(new_feature_scaled).squeeze()

                    old_embedding = np.load('Product/features_pca.pkl', allow_pickle = True)
                    old_features = np.load('Product/features_np_2.pkl', allow_pickle = True)

                    topwear_features = old_features[0:187]
                    bottomwear_features = old_features[187 : 387]
                    footwear_features = old_features[387 : 590]
                    eyewear_features = old_features[591 : 792]
                    handbag_features = old_features[793 : 994]

                    base_index = 0


                    distances, indices = [], []


                    base_index = 0

                    if file == "Topwear.jpg" :
                        features = data_json[(data_json['category'] == 'topwear') & (data_json['sex']==sex)]['features'].tolist()


                        neighbors = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='euclidean').fit(list(topwear_features))
                        distances, indices = neighbors.kneighbors([new_feature_scaled])
                        base_index = 0
                        st.header("\n Topwear Recommendations \n")

                    elif file == "Bottomwear.jpg" :
                        features = data_json[(data_json['category'] == 'bottomwear') & (data_json['sex']==sex)]['features'].tolist()

                        neighbors = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='euclidean').fit(list(bottomwear_features))
                        distances, indices = neighbors.kneighbors([new_feature_scaled])
                        base_index = 187
                        st.header("\n Bottomwear Recommendations \n")

                    elif file == "Footwear.jpg" :
                        features = np.array(data_json[(data_json['category'] == 'footwear') & (data_json['sex']==sex)]['features'].tolist())



                        neighbors = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='euclidean').fit(footwear_features)
                        distances, indices = neighbors.kneighbors([new_feature_scaled])
                        base_index = 387
                        st.header("\n Footwear Recommendations \n")


                    elif file == "Handbag.jpg" :
                        features = data_json[data_json['category'] == 'handbag']['features'].tolist()

                        eighbors = NearestNeighbors(n_neighbors=15, algorithm='brute', metric='euclidean').fit(list(handbag_features))
                        distances, indices = neighbors.kneighbors([new_feature_scaled])
                        base_index = 792
                        st.header("\n Handbag Recommendations \n")

                    else:
                        pass



                    indexes = indices


                    root_dir = 'Product'

                    filenames = []
                    category = data_json.iloc[indices[0],:]['category']

                    names = []
                    prices = []
                    links = []

                    for index in indexes[0] :
                        if data_json.iloc[index + base_index,:]['sex']==sex :
                            img_path = os.path.join(root_dir, data_json.iloc[index + base_index,:]['category'], data_json.iloc[index + base_index,:]['name']) + '.jpg'
                            filenames.append(img_path)


                            links.append(data_json.iloc[index + base_index,:]['link'])
                            names.append(data_json.iloc[index + base_index, :]['name'])
                            prices.append(data_json.iloc[index + base_index, : ]['price'])

                        if len(links)>=5:
                            break

                        

                    counter = 0


                    index = 0
                    for img_path in filenames :
                        try :                        
                            img = Image.open(img_path)

                            index = filenames.index
                            name, link = names[counter], links[counter]
                            st.image(img, caption='test', width  = 200)
                            st.write(name)
                            st.write("Price : " +  str(prices[counter]) + " \u20B9")
                            st.write(link)
                            counter += 1
                            if counter>=5 :
                                break
                        except :
                            pass

                except :
                    print(file)

            clear_folder('temp')
            clear_folder('user_products')       
                 








    else:
        st.error("Not  full shot")

   












import streamlit as st
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

st.title('Dog Breed Classifier App ')
st.header('Made By Lakhan Singh :sunglasses:',divider='rainbow')

label = {'afghan_hound': 0,
 'african_hunting_dog': 1,
 'airedale': 2,
 'basenji': 3,
 'basset': 4,
 'beagle': 5,
 'bedlington_terrier': 6,
 'bernese_mountain_dog': 7,
 'black-and-tan_coonhound': 8,
 'blenheim_spaniel': 9,
 'bloodhound': 10,
 'bluetick': 11,
 'border_collie': 12,
 'border_terrier': 13,
 'borzoi': 14,
 'boston_bull': 15,
 'bouvier_des_flandres': 16,
 'brabancon_griffon': 17,
 'bull_mastiff': 18,
 'cairn': 19,
 'cardigan': 20,
 'chesapeake_bay_retriever': 21,
 'chow': 22,
 'clumber': 23,
 'cocker_spaniel': 24,
 'collie': 25,
 'curly-coated_retriever': 26,
 'dhole': 27,
 'dingo': 28,
 'doberman': 29,
 'english_foxhound': 30,
 'english_setter': 31,
 'entlebucher': 32,
 'flat-coated_retriever': 33,
 'german_shepherd': 34,
 'german_short-haired_pointer': 35,
 'golden_retriever': 36,
 'gordon_setter': 37,
 'great_dane': 38,
 'great_pyrenees': 39,
 'groenendael': 40,
 'ibizan_hound': 41,
 'irish_setter': 42,
 'irish_terrier': 43,
 'irish_water_spaniel': 44,
 'irish_wolfhound': 45,
 'japanese_spaniel': 46,
 'keeshond': 47,
 'kerry_blue_terrier': 48,
 'komondor': 49,
 'kuvasz': 50,
 'labrador_retriever': 51,
 'leonberg': 52,
 'lhasa': 53,
 'malamute': 54,
 'malinois': 55,
 'maltese_dog': 56,
 'mexican_hairless': 57,
 'miniature_pinscher': 58,
 'miniature_schnauzer': 59,
 'newfoundland': 60,
 'norfolk_terrier': 61,
 'norwegian_elkhound': 62,
 'norwich_terrier': 63,
 'old_english_sheepdog': 64,
 'otterhound': 65,
 'papillon': 66,
 'pekinese': 67,
 'pembroke': 68,
 'pomeranian': 69,
 'pug': 70,
 'redbone': 71,
 'rhodesian_ridgeback': 72,
 'rottweiler': 73,
 'saint_bernard': 74,
 'saluki': 75,
 'samoyed': 76,
 'schipperke': 77,
 'scotch_terrier': 78,
 'scottish_deerhound': 79,
 'sealyham_terrier': 80,
 'shetland_sheepdog': 81,
 'standard_poodle': 82,
 'standard_schnauzer': 83,
 'sussex_spaniel': 84,
 'tibetan_mastiff': 85,
 'tibetan_terrier': 86,
 'toy_terrier': 87,
 'vizsla': 88,
 'weimaraner': 89,
 'whippet': 90,
 'wire-haired_fox_terrier': 91,
 'yorkshire_terrier': 92}

model_summary = '''Model: "sequential"
================================================================
 Layer (type)                Output Shape              Param    
=================================================================
 random_flip (RandomFlip)    (None, 400, 400, 3)       0         
                                                                 
 random_rotation (RandomRot  (None, 400, 400, 3)       0         
 ation)                                                          
                                                                 
 keras_layer (KerasLayer)    (None, 1280)              20331360  
                                                                 
 dropout (Dropout)           (None, 1280)              0         
                                                                 
 dense (Dense)               (None, 93)                119133    
                                                                 
=================================================================
Total params: 20450493 (78.01 MB)
Trainable params: 119133 (465.36 KB)
Non-trainable params: 20331360 (77.56 MB)
================================================================='''


label_dt = pd.DataFrame(label,index=label.values())

if st.checkbox('Do you want to check all breeds of dog that is used in this model'):
    st.write('These are here')
    st.write(label_dt.head(1))

if st.checkbox('Check here model summary :sunglasses:'):
    st.text(model_summary)

img = st.file_uploader('## Upload a dog image to classified it breed : ',type=['png', 'jpg','jpeg'])

@st.cache_resource

def load_model():
    model = tf.keras.models.load_model('Dog_Breed_Classifier.h5',custom_objects={'KerasLayer': hub.KerasLayer})
    return model

model = load_model()

st.text('Model Loaded Suceessfully ...')

def predict(model, img):
    img = tf.keras.utils.load_img(img,target_size=(400,400))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array/255
    st.image(img_array)

    img_array = np.expand_dims(img_array,axis=0)
    predictions = model.predict(img_array)

    print('Prediction Value of the image is : ', np.argmax(predictions))

    predicted_class = [i for i ,j in label.items() if j == np.argmax(predictions)]
    confidence = round(100*(np.max(predictions[0])), 2)

    st.subheader(f" Predicted Class: {predicted_class[0]}")
    st.subheader(f" Confidence: {confidence}%")

if st.button('show image with prediction'):
    result = predict(model , img)

if st.button("Clear All Cache"):
    st.cache_data.clear()


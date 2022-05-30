import streamlit as st
import cv2
import os
import matplotlib.pyplot as plt
import reverseImgSearchMilvus as rism


res_arr = []
res_imgs_col_1 = None
res_imgs_col_2 = None
db_info = []

       
def test_imgs_button_callback(id, img_dict):
    global res_arr,db_info
    img_path = img_dict[id]['path']
    

    #st.write("Label Matched to: {0}".format(reverseImgSearch.search(img_path)[1][0][0]))
    #res_arr= [['Venus_Williams', './data/lfw/Venus_Williams/Venus_Williams_0001.jpg']]*20
    img_db, fmt, id_to_label, train_paths = db_info
    res_arr = rism.search(img_path, img_db, fmt, id_to_label, train_paths)
    repopulate_res()
    
def repopulate_res():
    global res_arr, res_imgs_col_1, res_imgs_col_2
    for index, res in enumerate(res_arr[:10]):
        img = cv2.imread(res[1])
        with res_imgs_col_1[index]:
            st.image(img, channels='BGR')
            st.write("{0}".format(index+1))
            st.write("{0}".format(res[0].split("_")[0]))
            if len(res[0].split("_"))>1:
                st.write("{0}".format(res[0].split("_")[1]))
            
    for index, res in enumerate(res_arr[10:]):
        img = cv2.imread(res[1])
        with res_imgs_col_2[index]:
            st.image(img, channels='BGR')
            st.write("{0}".format(index+11))
            st.write("{0}".format(res[0].split("_")[0]))
            if len(res[0].split("_"))>1:
                st.write("{0}".format(res[0].split("_")[1]))
    
def main():
    global res_imgs_col_1, res_imgs_col_2, db_info

    
    #img_db, fmt, id_to_label, train_paths= create_db()
    db_info = rism.create_db()
    
    test_img_files = os.listdir("./data/test_data/")

    st.write("Select one of the following test images to find similar images in the DB")
    test_imgs_cols = st.columns(10)
    
    test_img_button_dict = {}

    for index,file in enumerate(test_img_files):
        img = cv2.imread("./data/test_data/"+file)
        with test_imgs_cols[index]:
            st.image(img, channels='BGR')
            st.button("Test".format(index),key = "test_img_{0}_button".format(index), on_click=test_imgs_button_callback, kwargs = {"id":index, "img_dict": test_img_button_dict})
            test_img_button_dict[index] = {"path":"./data/test_data/"+file}
            
      
    st.write("Image Search Results:")
    res_imgs_col_1 = st.columns([4]*10)
    res_imgs_col_2 = st.columns(10)
    
    
    
    
    
    

main()
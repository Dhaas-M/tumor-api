import sys 
import cv2
import pandas as pd
from nanoid import generate
from keras.models import load_model
from PIL import Image
import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from skimage.color import rgb2ypbpr,rgbcie2rgb,rgb2lab,rgb2hed
from skimage.filters import meijering
import numpy as np
import joblib
from keras import Model
from sklearn.metrics import accuracy_score







def TestBT(In_Img):
        output = {}
        res_path = generate()
    
    # In_Img='pred16.jpg'

    # i = In_Img[23:34])

    # Img_Dis(In_Img)
        # print("BRAIN STAGES")
        # _Using_different_filters_to_show_images_in_various_views
        i1 = imread(In_Img)
        img = imread(In_Img)
        i = imread(In_Img)
        img = img - 100.000
        # img_new = meijering(img)
        img1 = meijering(img)
        img_1 = rgbcie2rgb(img1) - 1
        img2 = rgb2lab(img)
        #
        plt.plot(), imshow(i1)
        plt.title("Input Image")
        # plt.axis('off')
        plt.savefig('static/images/'+res_path+'1.png')
        output['img1'] = 'static/images/'+res_path+'1.png'
        
        

        plt.subplot(121), imshow(i)
        plt.title("input_image")
        # plt.axis('off')

        plt.subplot(122), imshow(img)
        plt.title("View-1")
        # plt.axis('off')
        plt.savefig('static/images/'+res_path+'2.png')
        output['img2'] = 'static/images/'+res_path+'2.png'
    

        plt.subplot(121), imshow(img2)
        plt.title("View-2")

        plt.subplot(122), imshow(img_1)
        plt.title("View-3")
        # plt.axis('off')
        plt.savefig('static/images/'+res_path+'3.png')
        output['img3'] = 'static/images/'+res_path+'3.png'

        # plt.subplot(121), imshow(img_new)
        # plt.title("View-4")
        # plt.title('RGB Format')

        # plt.subplot(122), imshow(img_new)
        # plt.title('HSV Format')
        # Img_cr(In_Img)
        model = load_model('static\models\Brain-CNN.h5')
        batch_size = 10
        image = cv2.imread(In_Img)
        img = Image.fromarray(image)
        img = img.resize((100, 100))
        img = np.array(img)
        input_img = np.expand_dims(img, axis=0)
        # print(input_img)
        # print(input_img.shape)
        # result = model.predict_classes(input_img)
        result = model.predict(input_img)
        # print(result)
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.output)
        i = intermediate_layer_model.predict(input_img)
        # print("in_m",i)
        # Load the SVM_model from the file
        svm_model = joblib.load('static\models\Brain-Conv-SVM.pkl')
        # Use the loaded model to make predictions
        r = svm_model.predict(i)
        # print(r[0])

        # Load the SVM_model from the file
        xgb_model = joblib.load('static\models\Brain-Conv-XGB.pkl')
        # Use the loaded model to make predictions
        r1 = xgb_model.predict(i)
        
        output["ans"] = result.item(0)
        return  output
        # print(r1[0])
        # svm-model = load_model('D:\Conv-SVM.pkl')
        # r = svm-model.predict()
        # print(r)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(blur, 10, 150, 3)
        dilated = cv2.dilate(canny, (1, 1), iterations=0)

        (cnt, hierarchy) = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb, cnt, -1, (0, 256, 0), 0)
        print(len(cnt))
        # print("TUMOR STATUS:")
        # Tumor_present_print_[[0]]
        # Tumor_absent_print_[[1]]



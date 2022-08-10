import base64
import string
from turtle import color, width
import streamlit as st
import hydralit_components as hc
import datetime
import pandas as pd
import io
import re
from PIL import Image
from streamlit_option_menu import option_menu
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model,load_model
from keras.layers import Conv2D, Input, add
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from skimage.measure import compare_ssim
import os
import cv2
import h5py
import numpy 
from skimage import io
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from htbuilder import HtmlElement, div, br, hr, a, p, img, styles
from htbuilder.units import percent,px





def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """

    style_div = styles(
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        text_align="center",
        height="60px",
        opacity=0.5
    )

    style_hr = styles(
    )

    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer():
    myargs = [
        "<b> Designed by : ❤️ ",
        link("https://www.linkedin.com/in/ossamajali/", "Ossama Majali"),
        br(),
    ]
    layout(*myargs)


def MSEloss(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))
def dssimloss(y_true, y_pred):
    ssim2 = tf.image.ssim(y_true, y_pred, 1.0)
    return K.mean(1 - ssim2)
def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, 1.0)
def SR(image,model,scale):

    fig = plt.figure(figsize=(23, 23))
    ax1 = fig.add_subplot(1,3,1)
    ax1.set_title('Original' , fontsize=20, color= '#C8AD7F', fontweight='bold')
    ax2 = fig.add_subplot(1,3,2)
    ax2.set_title('Bicubic (Input)' , fontsize=20, color= '#C8AD7F', fontweight='bold')
    ax3 = fig.add_subplot(1,3,3)
    ax3.set_title('Super Resolution (Ouput)' , fontsize=20, color= '#C8AD7F', fontweight='bold')
    



    # original image 
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    eval_img1 = original
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    img = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
    shape = img.shape


    # Bicubic image 
    Y_img = cv2.resize(img[:, :, 0], (int(shape[1] / scale), int(shape[0] / scale)), cv2.INTER_CUBIC)
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    ax2.imshow( cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    eval_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[:, :, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    ax3.imshow( cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    eval_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.pyplot(fig)

    evalPSNR(eval_img1,eval_img2,eval_img3)
    

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
def psnr(im1, im2):
    img_arr1 = numpy.array(im1).astype('float32')
    img_arr2 = numpy.array(im2).astype('float32')
    mse = tf.math.reduce_mean(tf.math.squared_difference(img_arr1, img_arr2))
    psnr = tf.constant(255 ** 2, dtype=tf.float32) / mse
    result = tf.constant(10, dtype=tf.float32) * log10(psnr)
    return result

def evalPSNR(imp1,imp2,imp3):
    im1 = cv2.cvtColor(imp1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    im2 = cv2.cvtColor(imp2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    im3 = cv2.cvtColor(imp3, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    LR = psnr(im1, im2)
    SR = psnr(im1, im3)
 
    df = pd.DataFrame()
    header('Quality measures (PSNR / SSIM)')
    df = df.append({'Type': 'Low resolution', 'PSNR': tf.get_static_value("%.2f"%LR), 'SSIM': compare_ssim(im1, im2)}, ignore_index = True)
    df = df.append({'Type': 'Super resolution', 'PSNR': tf.get_static_value("%.2f"%SR), 'SSIM': compare_ssim(im1, im3)}, ignore_index = True)
    st.table(df)






def header(url):
     st.markdown(f'<p style="color:#C8AD7F;font-size:24px; font-weight: bold">{url}</p>', unsafe_allow_html=True)

if __name__ == '__main__': 
    menu_id = option_menu(
            menu_title=None,  # required
            options=["Home", "Super-Resolution"],  # required
            icons=["house", "bi-badge-hd"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
           styles={
                "nav-link-selected": {"background-color": "#C8AD7F"},
            },
        )
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            Header {visibility: hidden;}
            .stSpinner > div > div {border-top-color: #C8AD7F;}
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    if menu_id =='Home':
        st.title("Deep Learning-based super resolut-ion image reconstruction")
        st.write("Super Resolution is the process of improving the quality of a image by enhancing its apparent resolution. Having an algorithm that effectively imagines the detail that would be present if the image was at a higher resolution.")
        st.image("Images/sr.png")
        header("References")
        st.markdown('<p style="font-style: italic; font-family: \'Times New Roman\', monospace;">Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Learning a deep convolutional network for image super-resolution. In European conference on computer vision, pages 184-199. Springer, 2014.</p>', unsafe_allow_html=True)
    elif menu_id =="Super-Resolution":
        header("Select an image")
        img_file = st.file_uploader("Upload File", type=['png', 'jpg', 'jpeg','bmp'])
        col1, col2, col3 = st.columns(3)
        col1.write(' ')
        col3.write(' ')
        if img_file is not None:
            up_img = io.imread(img_file)
            header('Pre-processing')
            #col2.image(up_img)
            listSearch= ['x2','x3','x4','x8']
            __1,__2,__3=st.columns((3))
            opt_tick=__2.selectbox("Select the upscale factor",options=listSearch)     
            if opt_tick == 'x2':
                scale = 2
                s = str(opt_tick)
            elif opt_tick == 'x3':
                scale = 3
                s = str(opt_tick)
            elif opt_tick == 'x4':
                scale = 4
                s = str(opt_tick)
            elif opt_tick == 'x8':   
                scale = 8
                s = str(opt_tick)
            else:
                scale = 2
                s = "x2"
            model = tf.keras.models.load_model("model-100epoch-"+s+".h5", custom_objects={'dssimloss': dssimloss, 'PSNR': PSNR, 'SSIM': SSIM})
            
            with st.spinner('Wait image processing in progress ..'):
                SR(up_img,model,scale) 
    footer()
           
        




     













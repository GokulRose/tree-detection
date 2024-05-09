import tensorflow as tf
import keras
import numpy as np

def load_and_prep_image(filename, img_shape=512):
  """
  Reads an image from filename, turns it into a tensor and reshapes it
  to (img_shape, img_shape, colour_channels).
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode the read file into a tensor
  img = tf.image.decode_image(img)
  # Resize the image
  img = tf.image.resize(img, size=[img_shape, img_shape])
  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img

import matplotlib.pyplot as plt

def pred_and_plot(model, filename, class_names=["Mango","Bamboo","Banana","Coconut","Papaya"]):
  """
  Imports an image located at filename, makes a prediction with model
  and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  result = model.predict(tf.expand_dims(img, axis=0))
  print(type(result))

  # Add in logic for multi-class & get pred_class name
  a = np.round(result[0][0])
  b = np.round(result[0][1])
  c = np.round(result[0][2])
  d = np.round(result[0][3])
  e = np.round(result[0][4])
  print(a,b,c,d,e)

  if a==1:
    pred_class="Mango"
    str1='''MANGO

Scientific name - Mangifera indica

Uses -
•Boost the Immune System.
•Improve Skin Health.
•May Ease Constipation.
•Support Eye Health'''
          
  elif b==1:
    pred_class ="Bamboo"
    str1='''BAMBOO
Scientific name - Bambusa vulgaris

Uses
•Building material
•Furniture
•Food
•paper'''

  elif c==1:
    pred_class="Banana"
    str1='''PLANTAIN (BANANA) 

Scientific name - Musa paradisiaca. 

Uses

•natural food wrappers,
 •eco-friendly plates,
•serving vessels'''

  elif d==1:
    pred_class="Coconut"
    str1='''COCONUT
Scientific name - Cocos nucifera

Uses
•Flesh: Milk, Food & Flour.
•Flowers: Medicine.
•Husks: Ropes.'''

  elif e==1:
    pred_class="Papaya"
    str1='''PAPAYA 

Scientific name - Carica papaya

Uses 
•Reduces inflammation
•Antioxidant powerhouse
'''

  else:
    pred_class="unknown"
    str1=" "

  print(pred_class)
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class} Prediction percentage :{int(result.max()*100)}")
  plt.axis(False);
  plt.show()
  print(str1)


model_1=keras.models.load_model('rahul.h5')
import os
path = 'test'
files = []
print(path)
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     files.append(os.path.join(r, file))
for f in files:
  pred_and_plot(model=model_1,filename=f)

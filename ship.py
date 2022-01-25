#%%
import os 
import json
import numpy as np
import xml.etree.ElementTree as elemTree
import tensorflow as tf
from PIL import Image

#%%
def serialize_example(dic):
    image = dic["image"].tobytes()
    image_shape = np.array(dic["image_shape"]).tobytes()
    bbox = dic["bbox"].tobytes()
    bbox_shape = np.array(dic["bbox_shape"]).tobytes()
    label = dic["label"].tobytes()

    dic = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'image_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_shape])),
        'bbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox])),
        'bbox_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bbox_shape])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
    })) 
    return dic.SerializeToString()

#%%
def deserialize_example(serialized_string):
    image_feature_description = { 
        'image': tf.io.FixedLenFeature([], tf.string), 
        'image_shape': tf.io.FixedLenFeature([], tf.string), 
        'bbox': tf.io.FixedLenFeature([], tf.string), 
        'bbox_shape': tf.io.FixedLenFeature([], tf.string), 
        'label': tf.io.FixedLenFeature([], tf.string), 
    } 
    example = tf.io.parse_single_example(serialized_string, image_feature_description) 

    image = tf.io.decode_raw(example["image"], tf.float32)
    image_shape = tf.io.decode_raw(example["image_shape"], tf.int32)
    bbox = tf.io.decode_raw(example["bbox"], tf.float32)
    bbox_shape = tf.io.decode_raw(example["bbox_shape"], tf.int32)
    label = tf.io.decode_raw(example["label"], tf.int32) 

    image = tf.reshape(image, image_shape)
    bbox = tf.reshape(bbox, bbox_shape)
    
    return image, bbox, label
#%%
# filename_lst = []

# save_dir = r"D:\won\ship_tfrecord"
# name = 'train'
# writer = tf.io.TFRecordWriter(f'{save_dir}\{name}.tfrecord'.encode("utf-8"))

# work_dir = r"C:\won\data\ship_detection\train\남해_여수항1구역_BOX"
# work_contents = os.listdir(work_dir)
# for i in range(len(work_contents)):
#     path_dir = work_dir + r"\\" + work_contents[i]
#     path_contents = os.listdir(path_dir)
#     for j in range(len(path_contents)):
#         data_dir = path_dir + r"\\" + path_contents[j]
#         data_contents = os.listdir(data_dir)
#         filename_lst = list(set([data_contents[l][:25] for l in range(len(data_contents))]))

#         for k in range(len(filename_lst)):
#             file_dir = data_dir + r"\\" + filename_lst[k]
#             filename = filename_lst[k]
#             # filename_lst.append(filename)

#             #jpg
#             img_ = Image.open(file_dir + ".jpg")
#             img_ = tf.convert_to_tensor(np.array(img_, dtype=np.int32)) / 255 # image
#             img_ = tf.image.resize(img_, (432, 768))
#             img = np.array(img_)

#             #xml
#             tree = elemTree.parse(file_dir + ".xml")
#             root = tree.getroot()
#             bboxes_ = []
#             labels_ = []
#             for x in root:
#                 # print(x.tag)
#                 if x.tag == "object":
#                     for y in x:
#                         # print("--", y.tag)
#                         if y.tag == "bndbox":
#                             bbox_ = [int(z.text) for z in y] 
#                             bbox = [bbox_[0] / 2160, bbox_[1] / 3840, bbox_[2] / 2160, bbox_[3] / 3840]
#                             # print("----", bbox)
#                             bboxes_.append(bbox)
#                         if y.tag == "category_id":
#                             # print("----", y.text)
#                             labels_.append(int(y.text))
#             bboxes = np.array(bboxes_, dtype=np.float32)
#             labels = np.array(labels_, dtype=np.int32)

#             #json
#             with open(file_dir + "_meta.json", "r", encoding="UTF8") as st_json:
#                 st_python = json.load(st_json)
#             st_python["Date"]
#             time = st_python["Date"][11:-1]
#             weather = st_python["Weather"]
#             season = st_python["Season"]

#             #to_dictionary
#             dic = {
#                 "image":img,
#                 "image_shape":img.shape,
#                 "bbox":bboxes,
#                 "bbox_shape":bboxes.shape,
#                 "label":labels,
#             }

#             # info_ = {
#             #     "filename":filename,
#             #     "time":time,
#             #     "weather":weather,
#             #     "season":season,
#             #     "time":time,
#             #     "weather":weather,
#             #     "season":season
#             # }
#             # info = np.array([info_])

#             writer.write(serialize_example(dic))

#             # info_dir = r'C:\won\data\ship_detection\datasets\info\\' + filename
#             # np.save(info_dir + ".npy", info, allow_pickle=True)

# # filename_lst = np.array(filename_lst)
# # filename_dir = r'C:\won\data\ship_detection\datasets\filenames.npy'
# # np.save(filename_dir, filename_lst, allow_pickle=True)

# %%
#%%visualization
# import matplotlib.pyplot as plt
# from PIL import ImageDraw
# image = tf.keras.preprocessing.image.array_to_img(img)
# draw = ImageDraw.Draw(image)
# for bbox in bboxes:
#      x1, y1, x2, y2 = tf.split(bbox, 4, axis=-1)
#      x1, y1, x2, y2 = x1*432, y1*768, x2*432, y2*768
#      draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
# plt.figure()
# plt.imshow(image)
# plt.show()

# data_dir = r"C:\won\data\ship_detection"
# name = 'train'
# sample = tf.data.TFRecordDataset(f"{data_dir}/{name}.tfrecord".encode("utf-8")).map(deserialize_example)
# sample = sample.shuffle(buffer_size=30000, reshuffle_each_iteration=True)
# padding_values = (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
# data_shapes = ([None, None, None], [None, None], [None,])
# dataset = sample.padded_batch(4, data_shapes, padding_values, drop_remainder=True)
# for i in dataset.take(1):
#     break
# i
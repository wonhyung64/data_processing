#%%
import time
import data_utils
import tensorflow as tf

#%%
data_dir = r"C:/won/data/tfds"
train, validation, test, labels = data_utils.download_dataset("coco17", data_dir)
data_iter = iter(train)

#%%
save_dir = r"D:/won"
name = "test"
save_dir = save_dir + r"/data/" + "coco07" + "_tfrecord_" + str(416) + "_" + str(416)

try_num = 0
start_time = time.time()
writer = tf.io.TFRecordWriter(f"{save_dir}/{name}.tfrecord".encode("utf-8"))
while try_num < 100:
    sample = next(data_iter)
    image = sample["image"]/255
    bbox = sample["objects"]["bbox"]
    label = sample["objects"]["label"]
    not_diff = tf.logical_not(sample["objects"]["is_crowd"])
    bbox = bbox[not_diff]
    label = label[not_diff]
    example = {"image":image, "bbox":bbox, "label":label}
    x = data_utils.serialize_example(example, (416,416))
    writer.write(x)
    try_num += 1
    print(try_num)
print("total time :", time.time() - start_time)
print("try_num :" ,try_num)
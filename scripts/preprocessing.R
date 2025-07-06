library(reticulate)
use_python("C:/Users/admin/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe",
           required = TRUE)

library(keras)
library(tensorflow)

train_dir <- 'data/train'
val_dir <- 'data/valid'
test_dir <- 'data/test'

img_height <- 380
img_width <- 380
batch_size <- 64

train_datagen <- image_data_generator()

test_datagen <- image_data_generator()

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(img_height, img_width),
  batch_size = batch_size,
  class_mode = 'binary'
)

validation_generator <- flow_images_from_directory(
  val_dir,
  test_datagen,                     
  target_size   = c(img_height, img_width),
  batch_size    = batch_size,
  class_mode    = "binary"
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(img_height, img_width),
  batch_size = batch_size,
  class_mode = 'binary',
  shuffle = FALSE
)

# Note: All models now use this configuration:
# - 380x380 resolution
# - Batch size 64
# - No manual rescale (EfficientNet preprocessing will be applied)

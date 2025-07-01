library(reticulate)
use_python("C:/Users/Peter/AppData/Local/Programs/r-tf-venv/Scripts/python.exe",
           required = TRUE)

library(keras)
library(tensorflow)

train_dir <- 'data/train'
val_dir <- 'data/valid'
test_dir <- 'data/test'

img_height <- 224
img_width <- 224
batch_size <- 32

train_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 20,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  horizontal_flip = TRUE,
  zoom_range = 0.2
)

train_datagen <- image_data_generator(
  rescale            = 1/255,
  rotation_range     = 30,
  width_shift_range  = 0.2,
  height_shift_range = 0.2,
  brightness_range   = c(0.75,1.25),
  shear_range        = 0.2,
  horizontal_flip    = TRUE,
  channel_shift_range= 0.1,
  zoom_range         = 0.3,
  fill_mode          = "reflect"
)

test_datagen <- image_data_generator(
  rescale = 1/255
)

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

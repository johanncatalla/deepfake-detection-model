library(reticulate)
use_python("C:/Users/admin/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe",
           required = TRUE)

library(keras)
library(tensorflow)
library(fs)
library(magick)

train_dir <- 'data/train'
val_dir <- 'data/valid'
test_dir <- 'data/test'

img_height <- 380
img_width <- 380
batch_size <- 64

# EfficientNet preprocessing
keras_applications <- import("keras.applications.efficientnet")
efficientnet_preprocess <- keras_applications$preprocess_input

train_datagen <- image_data_generator(
  rotation_range = 20,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  shear_range = 0.1,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest",
  preprocessing_function = efficientnet_preprocess
)

test_datagen <- image_data_generator(
  preprocessing_function = efficientnet_preprocess
)

# Clean up 0kb and non-image files from original training directories
remove_0kb_files <- function(dir) {
  files <- list.files(dir, full.names = TRUE)
  sizes <- file.info(files)$size
  to_remove <- files[sizes == 0]
  if (length(to_remove) > 0) file.remove(to_remove)
}

keep_images <- function(dir) {
  files <- list.files(dir, full.names = TRUE)
  image_files <- files[grepl("\\.(jpg|jpeg|png)$", tolower(files))]
  to_remove <- setdiff(files, image_files)
  if (length(to_remove) > 0) file.remove(to_remove)
}

remove_invalid_images <- function(dir) {
  files <- list.files(dir, full.names = TRUE)
  for (f in files) {
    tryCatch({
      img <- image_read(f)
    }, error = function(e) {
      message("Removing invalid image: ", f)
      file.remove(f)
    })
  }
}

remove_0kb_files(file.path(train_dir, 'fake'))
remove_0kb_files(file.path(train_dir, 'real'))
keep_images(file.path(train_dir, 'fake'))
keep_images(file.path(train_dir, 'real'))
remove_invalid_images(file.path(train_dir, 'fake'))
remove_invalid_images(file.path(train_dir, 'real'))

# Use the original train_dir for the training generator
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
  batch_size    = batch_size %/% 2,
  class_mode    = "binary"
)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(img_height, img_width),
  batch_size = batch_size %/% 2,
  class_mode = 'binary',
  shuffle = FALSE
)

# Note: All models now use this configuration:
# - 380x380 resolution
# - Batch size 64
# - No manual rescale (EfficientNet preprocessing will be applied)

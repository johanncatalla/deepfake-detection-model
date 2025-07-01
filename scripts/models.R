library(keras)

focal_loss <- function(gamma = 2.0, alpha = 0.25) {
  function(y_true, y_pred) {
    # clip to prevent NaNs
    y_pred <- k_clip(y_pred, k_epsilon(), 1 - k_epsilon())
    # pt = y_pred if y_true==1 else 1-y_pred
    pt <- tf$where(tf$equal(y_true, 1), y_pred, 1 - y_pred)
    # compute the focal loss
    loss <- - alpha * k_pow((1 - pt), gamma) * k_log(pt)
    k_mean(loss)
  }
}

# -------------MODEL 1---------------------------------

# 1) Prepare the ResNet50 base (no top, frozen weights)
base_resnet <- application_resnet50(
  weights    = "imagenet",
  include_top= FALSE,
  input_shape= c(224,224,3)
)
freeze_weights(base_resnet)

# 2) Build your model by calling the base on an Input tensor
inputs <- layer_input(shape = c(224,224,3))

# call the base model as a function on inputs
x <- base_resnet(inputs)

# add your head  
x <- x %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.3) %>%
  layer_dense(1, activation = "sigmoid")

# instantiate the Model  
model_1 <- keras_model(inputs = inputs, outputs = x)

# 3) Compile  
model_1$compile(
  loss = focal_loss(gamma=2, alpha=0.25),
  optimizer = optimizer_adam(learning_rate = 1e-4),
  metrics   = list("accuracy")
)

# ----------------- MODEL 2: EfficientNetB0 -----------------

# 1) Base model
base_effnet <- application_efficientnet_b0(
  weights     = "imagenet",
  include_top = FALSE,
  input_shape = c(224,224,3)
)
freeze_weights(base_effnet)

# 2) Functional API head
inputs2 <- layer_input(shape = c(224,224,3))
x2      <- base_effnet(inputs2) %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.3) %>%
  layer_dense(1, activation = "sigmoid")

model_2 <- keras_model(inputs = inputs2, outputs = x2)

# 3) Compile via $compile()
model_2$compile(
  loss = focal_loss(gamma=2, alpha=0.25),
  optimizer = optimizer_adam(learning_rate = 1e-4),
  metrics   = list("accuracy")
)


# ----------------- MODEL 3: Lightweight CNN + SE -----------------

# 1) Define inputs
inputs3 <- layer_input(shape = c(224,224,3))

# 2) Body
body3 <- inputs3 %>%
  layer_conv_2d(32, c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(c(2,2)) %>%
  layer_conv_2d(64, c(3,3), activation = "relu") 

# 3) Global pooling
gap3 <- body3 %>%
  layer_global_average_pooling_2d()

# 4) Squeeze-and-Excitation (SE) block
se3 <- gap3 %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(64, activation = "sigmoid")

# 5) Recalibrate + head
out3 <- list(gap3, se3) %>%
  layer_multiply() %>%
  layer_dropout(0.3) %>%
  layer_dense(1, activation = "sigmoid")

model_3 <- keras_model(inputs = inputs3, outputs = out3)

# 6) Compile
model_3$compile(
  loss = focal_loss(gamma=2, alpha=0.25),
  optimizer = optimizer_adam(learning_rate = 1e-4),
  metrics   = list("accuracy")
)


# ----------------- MODEL 4: EffNetB0 + Dual Attention -----------------

# 1) Base
eff_base <- application_efficientnet_b0(
  weights     = "imagenet",
  include_top = FALSE,
  input_shape = c(224,224,3)
)
freeze_weights(eff_base)

# 2) Inputs & base output
inputs4 <- eff_base$input
x4      <- eff_base$output

# 3) Channel attention (SE)  
se4 <- x4 %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1280, activation = "sigmoid") %>%
  layer_reshape(target_shape = c(1,1,1280))

x4 <- layer_multiply(list(x4, se4))

# 4) Spatial attention (CBAM-like)
sa4 <- x4 %>%
  layer_conv_2d(filters     = 1,
                kernel_size = c(7,7),
                activation  = "sigmoid",
                padding     = "same")

x4 <- layer_multiply(list(x4, sa4))

# 5) Head
out4 <- x4 %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.3) %>%
  layer_dense(1, activation = "sigmoid")

model_4 <- keras_model(inputs = inputs4, outputs = out4)

# 6) Compile
model_4$compile(
  loss = focal_loss(gamma=2, alpha=0.25),
  optimizer = optimizer_adam(learning_rate = 1e-4),
  metrics   = list("accuracy")
)
library(keras)

# Note: used the following config for all models
# - 380x380 resolution
# - Binary crossentropy loss
# - 1e-3 learning rate
# - 0.5 dropout rate

# Focal loss implementation
focal_loss <- function(gamma = 2., alpha = 0.25) {
  function(y_true, y_pred) {
    epsilon <- k_epsilon()
    y_pred <- k_clip(y_pred, epsilon, 1.0 - epsilon)
    pt_1 <- tf$where(k_equal(y_true, 1), y_pred, k_ones_like(y_pred))
    pt_0 <- tf$where(k_equal(y_true, 0), y_pred, k_zeros_like(y_pred))
    loss <- -alpha * k_pow(1. - pt_1, gamma) * k_log(pt_1) - (1 - alpha) * k_pow(pt_0, gamma) * k_log(1. - pt_0)
    k_mean(loss)
  }
}

# -------------MODEL 1: ResNet50 ---------------------------------

base_resnet <- application_resnet50(
  weights    = "imagenet",
  include_top= FALSE,
  input_shape= c(380,380,3)  
)
freeze_weights(base_resnet)

inputs <- layer_input(shape = c(380,380,3))  

x <- base_resnet(inputs)

x <- x %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.5) %>%  # Changed to 0.5
  layer_dense(1, activation = "sigmoid")

model_1 <- keras_model(inputs = inputs, outputs = x)

model_1$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-3), 
  metrics   = list("accuracy")
)

# ----------------- MODEL 2: EfficientNetB0 -----------------

base_effnet <- application_efficientnet_b0(
  weights     = "imagenet",
  include_top = FALSE,
  input_shape = c(380,380,3)  
)
freeze_weights(base_effnet)

inputs2 <- layer_input(shape = c(380,380,3))  
x2      <- base_effnet(inputs2) %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.5) %>%  
  layer_dense(1, activation = "sigmoid")

model_2 <- keras_model(inputs = inputs2, outputs = x2)

model_2$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-3), 
  metrics   = list("accuracy")
)


# ----------------- MODEL 3: Lightweight CNN + SE  -----------------

inputs3 <- layer_input(shape = c(380,380,3))  

body3 <- inputs3 %>%
  layer_conv_2d(32, c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(c(2,2)) %>%
  layer_conv_2d(64, c(3,3), activation = "relu") 

gap3 <- body3 %>%
  layer_global_average_pooling_2d()

se3 <- gap3 %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(64, activation = "sigmoid")

out3 <- list(gap3, se3) %>%
  layer_multiply() %>%
  layer_dropout(0.5) %>% 
  layer_dense(1, activation = "sigmoid")

model_3 <- keras_model(inputs = inputs3, outputs = out3)

model_3$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-3),  
  metrics   = list("accuracy")
)


# ----------------- MODEL 4: EffNetB0 + Dual Attention -----------------

eff_base <- application_efficientnet_b0(
  weights     = "imagenet",
  include_top = FALSE,
  input_shape = c(380,380,3)  
)
freeze_weights(eff_base)

inputs4 <- eff_base$input
x4      <- eff_base$output

se4 <- x4 %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1280, activation = "sigmoid") %>%
  layer_reshape(target_shape = c(1,1,1280))

x4 <- layer_multiply(list(x4, se4))

sa4 <- x4 %>%
  layer_conv_2d(filters     = 1,
                kernel_size = c(7,7),
                activation  = "sigmoid",
                padding     = "same")

x4 <- layer_multiply(list(x4, sa4))

out4 <- x4 %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.5) %>% 
  layer_dense(1, activation = "sigmoid")

model_4 <- keras_model(inputs = inputs4, outputs = out4)

model_4$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-3),  
  metrics   = list("accuracy")
)

# ----------------- MODEL 5: EfficientNetB4 -----------------

base_effnetb4 <- application_efficientnet_b4(
  weights     = "imagenet",
  include_top = FALSE,
  input_shape = c(380, 380, 3)
)
freeze_weights(base_effnetb4)

inputs5 <- layer_input(shape = c(380, 380, 3))
x5 <- base_effnetb4(inputs5) %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")

model_efficientnetb4 <- keras_model(inputs = inputs5, outputs = x5)

model_efficientnetb4$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics = list("accuracy")
)

reduce_lr_effnetb4 <- callback_reduce_lr_on_plateau(
  monitor = "val_loss", 
  factor = 0.2, 
  patience = 2, 
  min_lr = 1e-6
)

# ----------------- MODEL 6: MobileNetV2 -----------------

base_mobilenetv2 <- application_mobilenet_v2(
  weights     = "imagenet",
  include_top = FALSE,
  input_shape = c(380, 380, 3)
)
freeze_weights(base_mobilenetv2)

inputs6 <- layer_input(shape = c(380, 380, 3))
x6 <- base_mobilenetv2(inputs6) %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")

model_mobilenetv2 <- keras_model(inputs = inputs6, outputs = x6)

model_mobilenetv2$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics = list("accuracy")
)
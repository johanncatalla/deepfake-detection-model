library(keras)

# Note: used the following config for all models
# - 380x380 resolution
# - Binary crossentropy loss
# - 1e-3 learning rate
# - 0.5 dropout rate

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
  loss = "binary_crossentropy", 
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
  loss = "binary_crossentropy",  
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
  loss = "binary_crossentropy", 
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
  loss = "binary_crossentropy",  
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
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics = list("accuracy")
)

reduce_lr_effnetb4 <- callback_reduce_lr_on_plateau(
  monitor = "val_loss", 
  factor = 0.2, 
  patience = 2, 
  min_lr = 1e-6
)
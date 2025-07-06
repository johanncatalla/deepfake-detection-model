if (!exists("class_weights")) {
  cat("Class weights not found. Calculating class weights...\n")
  tbl <- table(train_generator$classes)
  total <- sum(tbl)
  
  class_weights <- list(
    "0" = as.numeric(total / (2 * tbl["0"])),
    "1" = as.numeric(total / (2 * tbl["1"]))
  )
  print("Class weights calculated:")
  print(class_weights)
}

# ModelCheckpoint callback for fine-tuning
checkpoint_callback_fine <- callback_model_checkpoint(
  filepath = "best_model_fine_tuned.keras",
  monitor = "val_loss",
  save_best_only = TRUE,
  mode = "min",
  verbose = 1
)

n_fine_tune <- 50


#––––– 1) RESNET50 FINE-TUNING –––––#
total_layers <- length(base_resnet$layers)
freeze_until <- total_layers - n_fine_tune

for (i in seq_len(freeze_until)) {
  layer_i <- base_resnet$layers[[i]]
  py_set_attr(layer_i, "trainable", FALSE)
}
for (i in (freeze_until+1):total_layers) {
  layer_i <- base_resnet$layers[[i]]
  py_set_attr(layer_i, "trainable", TRUE)
}

# Focal loss implementation (must match models.R)
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

model_1$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-5),
  metrics   = list("accuracy")
)

steps_per_epoch   <- as.integer(ceiling(train_generator$n / batch_size))
validation_steps  <- as.integer(ceiling(validation_generator$n / batch_size))
n_fine_epochs     <- as.integer(15)

history_fine1 <- model_1$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_fine_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback_fine
  )
)

#––––– 2) EFFICIENTNETB0 (MODEL 2) FINE-TUNING  –––––#
total_layers <- length(base_effnet$layers)
freeze_until <- total_layers - n_fine_tune

for (i in seq_len(freeze_until)) {
  layer_i <- base_effnet$layers[[i]]
  py_set_attr(layer_i, "trainable", FALSE)
}
for (i in (freeze_until+1):length(base_effnet$layers)) {
  layer_i <- base_effnet$layers[[i]]
  py_set_attr(layer_i, "trainable", TRUE)
}

model_2$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-5),
  metrics   = list("accuracy")
)

#––– Fine-tune Model 2 –––#
history_fine2 <- model_2$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_fine_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback_fine
  )
)

#––––– 3) LIGHTWEIGHT CNN + SE (MODEL 3) FINE-TUNING –––––#

n_fine_tune   <- as.integer(30)
total_layers3 <- length(model_3$layers)

freeze_until3 <- as.integer(total_layers3 - n_fine_tune)
if (freeze_until3 < 0L) freeze_until3 <- 0L

for (i in seq_len(freeze_until3)) {
  layer_i <- model_3$layers[[i]]
  py_set_attr(layer_i, "trainable", FALSE)
}

start_unfreeze <- freeze_until3 + 1L
end_unfreeze   <- total_layers3
for (i in seq.int(start_unfreeze, end_unfreeze)) {
  layer_i <- model_3$layers[[i]]
  py_set_attr(layer_i, "trainable", TRUE)
}

model_3$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-5),
  metrics   = list("accuracy")
)

# Fine-tune Model 3
history_fine3 <- model_3$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_fine_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback_fine
  )
)

#––––– 4) DUAL-ATTENTION (MODEL 4) FINE-TUNING  –––––#

# --- Hyperparameters ---
n_fine_tune     <- as.integer(30)            

total_layers4  <- length(eff_base$layers)
freeze_until4  <- total_layers4 - n_fine_tune

for(i in seq_len(freeze_until4)) {
  layer_i <- eff_base$layers[[i]]
  py_set_attr(layer_i, "trainable", FALSE)
}

for(i in (freeze_until4+1):total_layers4) {
  layer_i <- eff_base$layers[[i]]
  py_set_attr(layer_i, "trainable", TRUE)
}

model_4$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-5),
  metrics   = list("accuracy")
)

history_fine4 <- model_4$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_fine_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor  = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback_fine
  )
)

# ----------------- Fine-tune EfficientNetB4 Model -----------------

# unfreeze the last 20 layers of the base model
fine_tune_at_effnetb4 <- -20

# unfreeze the selected layers for fine-tuning
for (i in seq_len(length(base_effnetb4$layers) + fine_tune_at_effnetb4)) {
  layer_i <- base_effnetb4$layers[[i]]
  py_set_attr(layer_i, "trainable", FALSE)
}
for (i in (length(base_effnetb4$layers) + fine_tune_at_effnetb4 + 1):length(base_effnetb4$layers)) {
  layer_i <- base_effnetb4$layers[[i]]
  # don't unfreeze batch norm layers
  if (!inherits(layer_i, "BatchNormalization")) {
    py_set_attr(layer_i, "trainable", TRUE)
  }
}

# specify a lower learning rate for the fine-tuned layers
optimizer_fine <- optimizer_adam(learning_rate = 1e-5)
model_efficientnetb4$compile(
  loss = focal_loss(),
  optimizer = optimizer_fine,
  metrics = list("accuracy")
)

n_fine_epochs_effnetb4 <- as.integer(15)

# train the model with fine-tuning
history_fine_effnetb4 <- model_efficientnetb4$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_fine_epochs_effnetb4,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks        = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback_fine
  )
)

#––––– 5) MobileNetV2 FINE-TUNING –––––#
n_fine_tune_mobilenetv2 <- as.integer(30)
total_layers_mobilenetv2 <- length(base_mobilenetv2$layers)
freeze_until_mobilenetv2 <- total_layers_mobilenetv2 - n_fine_tune_mobilenetv2

for (i in seq_len(freeze_until_mobilenetv2)) {
  layer_i <- base_mobilenetv2$layers[[i]]
  py_set_attr(layer_i, "trainable", FALSE)
}
for (i in (freeze_until_mobilenetv2+1):total_layers_mobilenetv2) {
  layer_i <- base_mobilenetv2$layers[[i]]
  py_set_attr(layer_i, "trainable", TRUE)
}

model_mobilenetv2$compile(
  loss = focal_loss(),
  optimizer = optimizer_adam(learning_rate = 1e-5),
  metrics = list("accuracy")
)

history_fine_mobilenetv2 <- model_mobilenetv2$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_fine_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback_fine
  )
)
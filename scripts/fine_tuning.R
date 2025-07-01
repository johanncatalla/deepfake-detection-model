n_fine_tune <- 50    # tune the last 30 layers; adjust as you like


#––––– 1) RESNET50 FINE-TUNING –––––#
total_layers <- length(base_resnet$layers)
freeze_until <- total_layers - n_fine_tune

# 1a) Freeze up through layer `freeze_until`
for (i in seq_len(freeze_until)) {
  layer_i <- base_resnet$layers[[i]]
  py_set_attr(layer_i, "trainable", FALSE)
}
# 1b) Unfreeze the rest
for (i in (freeze_until+1):total_layers) {
  layer_i <- base_resnet$layers[[i]]
  py_set_attr(layer_i, "trainable", TRUE)
}

# 2) Re-compile at a lower learning rate
model_1$compile(
  loss = focal_loss(gamma=2, alpha=0.25),
  optimizer = optimizer_adam(learning_rate = 1e-5),
  metrics   = list("accuracy")
)

# 3) Fine-tune

steps_per_epoch   <- as.integer(train_generator$n %/% batch_size)
validation_steps  <- as.integer(validation_generator$n %/% batch_size)
n_fine_epochs     <- as.integer(10)

# Then your fine-tune call becomes:
history_fine1 <- model_1$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_fine_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 3),
    callback_reduce_lr_on_plateau()
  )
)

#––––– 2) EFFICIENTNETB0 (MODEL 2) FINE-TUNING –––––#
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
  loss = focal_loss(gamma=2, alpha=0.25),
  optimizer = optimizer_adam(learning_rate = 1e-5),
  metrics   = list("accuracy")
)
steps_per_epoch   <- as.integer(train_generator$n %/% batch_size)
validation_steps  <- as.integer(validation_generator$n %/% batch_size)
n_fine_epochs     <- as.integer(10)    # or simply 10L

#––– Fine-tune Model 2 –––#
history_fine2 <- model_2$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_fine_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 3L),
    callback_reduce_lr_on_plateau()
  )
)

#––––– 3) LIGHTWEIGHT CNN + SE (MODEL 3) FINE-TUNING –––––#

n_fine_tune   <- as.integer(30)
total_layers3 <- length(model_3$layers)
# number of layers in model_3
total_layers3 <- length(model_3$layers)
# tune the last n_fine_tune layers just like the others
freeze_until3 <- as.integer(total_layers3 - n_fine_tune)
if (freeze_until3 < 0L) freeze_until3 <- 0L

# 4a) Freeze the early layers
for (i in seq_len(freeze_until3)) {
  layer_i <- model_3$layers[[i]]
  py_set_attr(layer_i, "trainable", FALSE)
}
# 4b) Unfreeze the last n_fine_tune layers
start_unfreeze <- freeze_until3 + 1L
end_unfreeze   <- total_layers3
for (i in seq.int(start_unfreeze, end_unfreeze)) {
  layer_i <- model_3$layers[[i]]
  py_set_attr(layer_i, "trainable", TRUE)
}

# 4c) Re-compile at a reduced learning rate
model_3$compile(
  loss = focal_loss(gamma=2, alpha=0.25),
  optimizer = optimizer_adam(learning_rate = 1e-5),
  metrics   = list("accuracy")
)
n_fine_epochs     <- as.integer(10)  
steps_per_epoch   <- as.integer(train_generator$n %/% batch_size)
validation_steps  <- as.integer(validation_generator$n %/% batch_size)
# 4d) Fine-tune
history_fine3 <- model_3$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_fine_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 3),
    callback_reduce_lr_on_plateau()
  )
)

#––––– 4) DUAL-ATTENTION (MODEL 4) FINE-TUNING –––––#

# --- Hyperparameters ---
n_fine_tune     <- as.integer(30)              # how many layers to unfreeze
n_fine_epochs   <- as.integer(10)              # fine-tune for 10 epochs
steps_per_epoch <- as.integer(train_generator$n %/% batch_size)
validation_steps<- as.integer(validation_generator$n %/% batch_size)

# --- 1) Freeze / Unfreeze Layers in eff_base ---
total_layers4  <- length(eff_base$layers)
freeze_until4  <- total_layers4 - n_fine_tune

# Freeze first (total_layers4 - n_fine_tune) layers
for(i in seq_len(freeze_until4)) {
  layer_i <- eff_base$layers[[i]]
  py_set_attr(layer_i, "trainable", FALSE)
}

# Unfreeze last n_fine_tune layers
for(i in (freeze_until4+1):total_layers4) {
  layer_i <- eff_base$layers[[i]]
  py_set_attr(layer_i, "trainable", TRUE)
}

# --- 2) Re-compile at Lower LR ---
model_4$compile(
  loss = focal_loss(gamma=2, alpha=0.25),
  optimizer = optimizer_adam(learning_rate = 1e-5),
  metrics   = list("accuracy")
)

# --- 3) Fine-tune ---
history_fine4 <- model_4$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_fine_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor  = "val_loss", patience = 3L),
    callback_reduce_lr_on_plateau()
  )
)
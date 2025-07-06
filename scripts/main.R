steps_per_epoch <- as.integer(ceiling(train_generator$n / batch_size))
validation_steps <- as.integer(ceiling(validation_generator$n / batch_size))

# Calculate class weights
tbl <- table(train_generator$classes)
total <- sum(tbl)
class_weights <- list(
  "0" = as.numeric(total / (2 * tbl["0"])),
  "1" = as.numeric(total / (2 * tbl["1"]))
)

# ModelCheckpoint callback
checkpoint_callback <- callback_model_checkpoint(
  filepath = "best_model_fc.keras",
  monitor = "val_loss",
  save_best_only = TRUE,
  mode = "min",
  verbose = 1
)

n_epochs <- as.integer(20) 

# Model 1: ResNet50 
history1 <- model_1$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback
  )
)

# Model 2: EfficientNetB0 
history2 <- model_2$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback
  )
)

# Model 3: Lightweight CNN + SE
history3 <- model_3$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback
  )
)

# Model 4: EffNetB0 + Dual Attention 
history4 <- model_4$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback
  )
)

# Model 5: EfficientNetB4 
history_effnetb4 <- model_efficientnetb4$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback
  )
)

# Model 6: MobileNetV2 
history_mobilenetv2 <- model_mobilenetv2$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 3, min_lr = 1e-7),
    checkpoint_callback
  )
)
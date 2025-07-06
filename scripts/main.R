steps_per_epoch <- as.integer(ceiling(train_generator$n / batch_size))
validation_steps <- as.integer(ceiling(validation_generator$n / batch_size))
n_epochs <- as.integer(5)  # changed to 5 epochs 

# Model 1: ResNet50 
history1 <- model_1$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 2),  
    callback_reduce_lr_on_plateau(factor = 0.2, patience = 2, min_lr = 1e-6) 
  )
)

# Model 2: EfficientNetB0 
history2 <- model_2$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 2), 
    callback_reduce_lr_on_plateau(factor = 0.2, patience = 2, min_lr = 1e-6) 
  )
)

# Model 3: Lightweight CNN + SE
history3 <- model_3$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 2),  
    callback_reduce_lr_on_plateau(factor = 0.2, patience = 2, min_lr = 1e-6)  
  )
)

# Model 4: EffNetB0 + Dual Attention 
history4 <- model_4$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 2),  
    callback_reduce_lr_on_plateau(factor = 0.2, patience = 2, min_lr = 1e-6)  
  )
)

# Model 5: EfficientNetB4 
history_effnetb4 <- model_efficientnetb4$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 2),
    callback_reduce_lr_on_plateau(factor = 0.2, patience = 2, min_lr = 1e-6)
  )
)
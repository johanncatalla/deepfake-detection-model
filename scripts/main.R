steps_per_epoch <- as.integer(ceiling(train_generator$n / batch_size))
validation_steps <- as.integer(ceiling(validation_generator$n / batch_size))
n_epochs          <- as.integer(20)

# Model 1
history1 <- model_1$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau()
  )
)

steps_per_epoch <- as.integer(ceiling(train_generator$n / batch_size))
validation_steps <- as.integer(ceiling(validation_generator$n / batch_size))
n_epochs         <- as.integer(20)

# Model 2
history2 <- model_2$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau()
  )
)

# Model 3
history3 <- model_3$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau()
  )
)

# Model 4
history4 <- model_4$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau()
  )
)

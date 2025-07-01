# 1) Compute class weights
tbl <- table(train_generator$classes)
total <- sum(tbl)

class_weights <- list(
  "0" = as.numeric(total / (2 * tbl["0"])),
  "1" = as.numeric(total / (2 * tbl["1"]))
)
print(class_weights)
# e.g. $`0` = (412+327)/(2*412) ≃ 0.892,  $`1` = (412+327)/(2*327) ≃ 1.122

# 2) Re–train each model with class_weight
steps_per_epoch  <- as.integer(train_generator$n %/% batch_size)
validation_steps <- as.integer(validation_generator$n %/% batch_size)
n_epochs         <- as.integer(20)

# Model 1
history1 <- model_1$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau()
  )
)

# Model 2
history2 <- model_2$fit(
  train_generator,
  steps_per_epoch  = steps_per_epoch,
  epochs           = n_epochs,
  validation_data  = validation_generator,
  validation_steps = validation_steps,
  class_weight     = class_weights,
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
  class_weight     = class_weights,
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
  class_weight     = class_weights,
  callbacks = list(
    callback_early_stopping(monitor = "val_loss", patience = 5),
    callback_reduce_lr_on_plateau()
  )
)

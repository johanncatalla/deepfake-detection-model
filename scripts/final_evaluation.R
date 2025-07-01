# Make sure test_steps is integer
test_steps <- as.integer(ceiling(test_generator$n / batch_size))

# Evaluate model_1 after fine-tuning
final_eval1 <- model_1$evaluate(
  test_generator,
  steps = test_steps
)
final_acc1 <- as.numeric(final_eval1[[2]])
cat("Model 1 final test accuracy:", round(final_acc1,4), "\n")

# And for the others...
final_eval2 <- model_2$evaluate(test_generator, steps = test_steps)
final_acc2 <- as.numeric(final_eval2[[2]])
cat("Model 2 final test accuracy:", round(final_acc2,4), "\n")

final_eval3 <- model_3$evaluate(test_generator, steps = test_steps)
final_acc3 <- as.numeric(final_eval3[[2]])
cat("Model 3 final test accuracy:", round(final_acc3,4), "\n")

final_eval4 <- model_4$evaluate(test_generator, steps = test_steps)
final_acc4 <- as.numeric(final_eval4[[2]])
cat("Model 4 final test accuracy:", round(final_acc4,4), "\n")


# Generate new predictions
pred_probs1_ft <- model_1$predict(test_generator, steps = test_steps)
pred_probs2_ft <- model_2$predict(test_generator, steps = test_steps)
pred_probs3_ft <- model_3$predict(test_generator, steps = test_steps)
pred_probs4_ft <- model_4$predict(test_generator, steps = test_steps)

# Then feed pred_probs*_ft and your true_labels into
# tune_and_eval() or your confusion/AUC code exactly as before.
res1_ft <- tune_and_eval(true_labels1, pred_probs1_ft)
res2_ft <- tune_and_eval(true_labels1, pred_probs2_ft)
res3_ft <- tune_and_eval(true_labels1, pred_probs3_ft)
res4_ft <- tune_and_eval(true_labels1, pred_probs4_ft)

# Summarize
summary_ft <- data.frame(
  Model     = c("ResNet50","EffNetB0","Lightweight+SE","DualAttention"),
  Threshold = c(res1_ft$threshold, res2_ft$threshold, res3_ft$threshold, res4_ft$threshold),
  Accuracy  = c(res1_ft$accuracy,   res2_ft$accuracy,   res3_ft$accuracy,   res4_ft$accuracy),
  Precision = c(res1_ft$precision,  res2_ft$precision,  res3_ft$precision,  res4_ft$precision),
  Recall    = c(res1_ft$recall,     res2_ft$recall,     res3_ft$recall,     res4_ft$recall),
  F1        = c(res1_ft$F1,         res2_ft$F1,         res3_ft$F1,         res4_ft$F1),
  AUC       = c(res1_ft$AUC,        res2_ft$AUC,        res3_ft$AUC,        res4_ft$AUC)
)
print(summary_ft)
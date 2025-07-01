library(caret)
library(pROC)

test_steps <- as.integer(ceiling(test_generator$n / batch_size))

eval_results1 <- model_1$evaluate(
  test_generator,
  steps = test_steps
)
acc1 <- as.numeric(eval_results1[[2]])

pred_probs1 <- model_1$predict(
  test_generator,
  steps = test_steps
)

pred_labels1 <- ifelse(pred_probs1 > 0.5, 1L, 0L)

true_labels1 <- test_generator$classes[1:length(pred_labels1)]


conf_mat1 <- confusionMatrix(
  factor(pred_labels1),
  factor(true_labels1),
  positive = "1"
)

roc1 <- roc(true_labels1, as.numeric(pred_probs1))
auc1 <- auc(roc1)

print(eval_results1)
print(conf_mat1)
cat("AUC:", round(auc1, 4), "\n")

#------------------MODEL 2------------------------------------------
eval_results2 <- model_2$evaluate(test_generator, steps = test_steps)
acc2 <- as.numeric(eval_results2[[2]])
pred_probs2   <- model_2$predict (test_generator, steps = test_steps)
pred_labels2  <- ifelse(pred_probs2 > 0.5, 1L, 0L)
conf_mat2     <- confusionMatrix(factor(pred_labels2), factor(true_labels1), positive = "1")
roc2          <- roc(true_labels1, as.numeric(pred_probs2))
auc2          <- auc(roc2)

#------------------MODEL 3------------------------------------------
eval_results3 <- model_3$evaluate(test_generator, steps = test_steps)
acc3 <- as.numeric(eval_results3[[2]])
pred_probs3   <- model_3$predict (test_generator, steps = test_steps)
pred_labels3  <- ifelse(pred_probs3 > 0.5, 1L, 0L)
conf_mat3     <- confusionMatrix(factor(pred_labels3), factor(true_labels1), positive = "1")
roc3          <- roc(true_labels1, as.numeric(pred_probs3))
auc3          <- auc(roc3)

#------------------MODEL 4------------------------------------------
eval_results4 <- model_4$evaluate(test_generator, steps = test_steps)
acc4 <- as.numeric(eval_results4[[2]])
pred_probs4   <- model_4$predict (test_generator, steps = test_steps)
pred_labels4  <- ifelse(pred_probs4 > 0.5, 1L, 0L)
conf_mat4     <- confusionMatrix(factor(pred_labels4), factor(true_labels1), positive = "1")
roc4          <- roc(true_labels1, as.numeric(pred_probs4))
auc4          <- auc(roc4)

#---------------------------------------------------------------------

fix_na0 <- function(x) if (is.na(x)) 0 else unname(x)

prec1 <- fix_na0(conf_mat1$byClass["Precision"])
prec2 <- fix_na0(conf_mat2$byClass["Precision"])
prec3 <- fix_na0(conf_mat3$byClass["Precision"])
prec4 <- fix_na0(conf_mat4$byClass["Precision"])

recall1 <- fix_na0(conf_mat1$byClass["Recall"])
recall2 <- fix_na0(conf_mat2$byClass["Recall"])
recall3 <- fix_na0(conf_mat3$byClass["Recall"])
recall4 <- fix_na0(conf_mat4$byClass["Recall"])

f11 <- fix_na0(conf_mat1$byClass["F1"])
f12 <- fix_na0(conf_mat2$byClass["F1"])
f13 <- fix_na0(conf_mat3$byClass["F1"])
f14 <- fix_na0(conf_mat4$byClass["F1"])


#-------------------- SUMMARY TABLE----------------------------------
results_summary <- data.frame(
  Model     = c("ResNet50", "EffNetB0", "Lightweight+SE", "DualAttention"),
  Accuracy  = c(acc1, acc2, acc3, acc4),
  Precision = c(prec1, prec2, prec3, prec4),
  Recall    = c(recall1, recall2, recall3, recall4),
  F1        = c(f11,  f12,  f13,  f14),
  AUC       = c(auc1,  auc2,  auc3,  auc4)
)
print(results_summary)


source("scripts/preprocessing.R")
library(caret)
library(pROC)

test_steps <- as.integer(ceiling(test_generator$n / batch_size))

eval_results1 <- model_1$evaluate(
  test_generator,
  steps = test_steps
)
acc1 <- as.numeric(eval_results1[[2]])

test_generator$reset()
pred_probs1 <- model_1$predict(test_generator, steps = test_steps)
pred_probs1 <- pred_probs1[1:test_generator$n]
true_labels1 <- test_generator$classes[1:length(pred_probs1)]
roc1 <- roc(true_labels1, as.numeric(pred_probs1))
opt_thr1 <- coords(roc1, "best", ret = "threshold", transpose = FALSE)
pred_labels1 <- ifelse(pred_probs1 > as.numeric(opt_thr1), 1L, 0L)
conf_mat1 <- confusionMatrix(factor(pred_labels1), factor(true_labels1), positive = "1")
auc1 <- auc(roc1)

print(eval_results1)
print(conf_mat1)
cat("AUC:", round(auc1, 4), "\n")

#------------------MODEL 2------------------------------------------
eval_results2 <- model_2$evaluate(test_generator, steps = test_steps)
acc2 <- as.numeric(eval_results2[[2]])

test_generator$reset()
pred_probs2   <- model_2$predict(test_generator, steps = test_steps)
pred_probs2 <- pred_probs2[1:test_generator$n]
true_labels2 <- test_generator$classes[1:length(pred_probs2)]
roc2 <- roc(true_labels2, as.numeric(pred_probs2))
opt_thr2 <- coords(roc2, "best", ret = "threshold", transpose = FALSE)
pred_labels2 <- ifelse(pred_probs2 > as.numeric(opt_thr2), 1L, 0L)
conf_mat2 <- confusionMatrix(factor(pred_labels2), factor(true_labels2), positive = "1")
auc2 <- auc(roc2)

#------------------MODEL 3------------------------------------------
eval_results3 <- model_3$evaluate(test_generator, steps = test_steps)
acc3 <- as.numeric(eval_results3[[2]])

test_generator$reset()
pred_probs3   <- model_3$predict(test_generator, steps = test_steps)
pred_probs3 <- pred_probs3[1:test_generator$n]
true_labels3 <- test_generator$classes[1:length(pred_probs3)]
roc3 <- roc(true_labels3, as.numeric(pred_probs3))
opt_thr3 <- coords(roc3, "best", ret = "threshold", transpose = FALSE)
pred_labels3 <- ifelse(pred_probs3 > as.numeric(opt_thr3), 1L, 0L)
conf_mat3 <- confusionMatrix(factor(pred_labels3), factor(true_labels3), positive = "1")
auc3 <- auc(roc3)

#------------------MODEL 4------------------------------------------
eval_results4 <- model_4$evaluate(test_generator, steps = test_steps)
acc4 <- as.numeric(eval_results4[[2]])

test_generator$reset()
pred_probs4   <- model_4$predict(test_generator, steps = test_steps)
pred_probs4 <- pred_probs4[1:test_generator$n]
true_labels4 <- test_generator$classes[1:length(pred_probs4)]
roc4 <- roc(true_labels4, as.numeric(pred_probs4))
opt_thr4 <- coords(roc4, "best", ret = "threshold", transpose = FALSE)
pred_labels4 <- ifelse(pred_probs4 > as.numeric(opt_thr4), 1L, 0L)
conf_mat4 <- confusionMatrix(factor(pred_labels4), factor(true_labels4), positive = "1")
auc4 <- auc(roc4)

#------------------MODEL 5 (EfficientNetB4)------------------------------------------
eval_results5 <- model_efficientnetb4$evaluate(test_generator, steps = test_steps)
acc5 <- as.numeric(eval_results5[[2]])

test_generator$reset()
pred_probs5   <- model_efficientnetb4$predict(test_generator, steps = test_steps)
pred_probs5 <- pred_probs5[1:test_generator$n]
true_labels5 <- test_generator$classes[1:length(pred_probs5)]
roc5 <- roc(true_labels5, as.numeric(pred_probs5))
opt_thr5 <- coords(roc5, "best", ret = "threshold", transpose = FALSE)
pred_labels5 <- ifelse(pred_probs5 > as.numeric(opt_thr5), 1L, 0L)
conf_mat5 <- confusionMatrix(factor(pred_labels5), factor(true_labels5), positive = "1")
auc5 <- auc(roc5)

#------------------MODEL 6 (MobileNetV2)------------------------------------------
eval_results6 <- model_mobilenetv2$evaluate(test_generator, steps = test_steps)
acc6 <- as.numeric(eval_results6[[2]])

test_generator$reset()
pred_probs6   <- model_mobilenetv2$predict(test_generator, steps = test_steps)
pred_probs6 <- pred_probs6[1:test_generator$n]
true_labels6 <- test_generator$classes[1:length(pred_probs6)]
roc6 <- roc(true_labels6, as.numeric(pred_probs6))
opt_thr6 <- coords(roc6, "best", ret = "threshold", transpose = FALSE)
pred_labels6 <- ifelse(pred_probs6 > as.numeric(opt_thr6), 1L, 0L)
conf_mat6 <- confusionMatrix(factor(pred_labels6), factor(true_labels6), positive = "1")
auc6 <- auc(roc6)

#---------------------------------------------------------------------

fix_na0 <- function(x) if (is.na(x)) 0 else unname(x)

prec1 <- fix_na0(conf_mat1$byClass["Precision"])
prec2 <- fix_na0(conf_mat2$byClass["Precision"])
prec3 <- fix_na0(conf_mat3$byClass["Precision"])
prec4 <- fix_na0(conf_mat4$byClass["Precision"])
prec5 <- fix_na0(conf_mat5$byClass["Precision"])
prec6 <- fix_na0(conf_mat6$byClass["Precision"])

recall1 <- fix_na0(conf_mat1$byClass["Recall"])
recall2 <- fix_na0(conf_mat2$byClass["Recall"])
recall3 <- fix_na0(conf_mat3$byClass["Recall"])
recall4 <- fix_na0(conf_mat4$byClass["Recall"])
recall5 <- fix_na0(conf_mat5$byClass["Recall"])
recall6 <- fix_na0(conf_mat6$byClass["Recall"])

f11 <- fix_na0(conf_mat1$byClass["F1"])
f12 <- fix_na0(conf_mat2$byClass["F1"])
f13 <- fix_na0(conf_mat3$byClass["F1"])
f14 <- fix_na0(conf_mat4$byClass["F1"])
f15 <- fix_na0(conf_mat5$byClass["F1"])
f16 <- fix_na0(conf_mat6$byClass["F1"])


#-------------------- SUMMARY TABLE----------------------------------
results_summary <- data.frame(
  Model     = c("ResNet50", "EffNetB0", "Lightweight+SE", "DualAttention", "EfficientNetB4", "MobileNetV2"),
  Accuracy  = c(acc1, acc2, acc3, acc4, acc5, acc6),
  Precision = c(prec1, prec2, prec3, prec4, prec5, prec6),
  Recall    = c(recall1, recall2, recall3, recall4, recall5, recall6),
  F1        = c(f11,  f12,  f13,  f14,  f15, f16),
  AUC       = c(auc1,  auc2,  auc3,  auc4,  auc5, auc6)
)
print(results_summary)
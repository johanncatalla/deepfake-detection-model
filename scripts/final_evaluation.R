library(pROC)
library(caret)

tune_and_eval <- function(true_labels, pred_probs, positive = 1L) {
  
  roc_obj <- roc(true_labels, as.numeric(pred_probs))
  
  opt <- coords(
    roc_obj, "best",
    ret = c("threshold","sensitivity","specificity"),
    transpose = FALSE
  )

  thr  <- as.numeric(opt["threshold"])
  sens <- as.numeric(opt["sensitivity"])
  spec <- as.numeric(opt["specificity"])
  
  cat(
    "Optimal threshold:", round(thr,3),
    "Sens:",        round(sens,3),
    "Spec:",        round(spec,3), "\n"
  )
  
  pred_labels_opt <- ifelse(pred_probs >= thr, positive, 1L - positive)

  cm <- confusionMatrix(
    factor(pred_labels_opt, levels = c(0,1)),
    factor(true_labels,     levels = c(0,1)),
    positive = as.character(positive)
  )
  
  auc_val <- as.numeric(auc(roc_obj))

  list(
    threshold = thr,
    accuracy  = unname(cm$overall["Accuracy"]),
    precision = unname(cm$byClass["Precision"]),
    recall    = unname(cm$byClass["Recall"]),
    F1        = unname(cm$byClass["F1"]),
    AUC       = auc_val,
    confusion = cm$table
  )
}

test_steps <- as.integer(ceiling(test_generator$n / batch_size))

final_eval1 <- model_1$evaluate(
  test_generator,
  steps = test_steps
)
final_acc1 <- as.numeric(final_eval1[[2]])
cat("Model 1 final test accuracy:", round(final_acc1,4), "\n")

final_eval2 <- model_2$evaluate(test_generator, steps = test_steps)
final_acc2 <- as.numeric(final_eval2[[2]])
cat("Model 2 final test accuracy:", round(final_acc2,4), "\n")

final_eval3 <- model_3$evaluate(test_generator, steps = test_steps)
final_acc3 <- as.numeric(final_eval3[[2]])
cat("Model 3 final test accuracy:", round(final_acc3,4), "\n")

final_eval4 <- model_4$evaluate(test_generator, steps = test_steps)
final_acc4 <- as.numeric(final_eval4[[2]])
cat("Model 4 final test accuracy:", round(final_acc4,4), "\n")

final_eval5 <- model_efficientnetb4$evaluate(test_generator, steps = test_steps)
final_acc5 <- as.numeric(final_eval5[[2]])
cat("Model 5 (EfficientNetB4) final test accuracy:", round(final_acc5,4), "\n")

pred_probs1_ft <- model_1$predict(test_generator, steps = test_steps)
pred_probs2_ft <- model_2$predict(test_generator, steps = test_steps)
pred_probs3_ft <- model_3$predict(test_generator, steps = test_steps)
pred_probs4_ft <- model_4$predict(test_generator, steps = test_steps)
pred_probs5_ft <- model_efficientnetb4$predict(test_generator, steps = test_steps)

res1_ft <- tune_and_eval(true_labels1, pred_probs1_ft)
res2_ft <- tune_and_eval(true_labels1, pred_probs2_ft)
res3_ft <- tune_and_eval(true_labels1, pred_probs3_ft)
res4_ft <- tune_and_eval(true_labels1, pred_probs4_ft)

true_labels5 <- true_labels1
res5_ft <- tune_and_eval(true_labels5, pred_probs5_ft)

summary_ft <- data.frame(
  Model     = c("ResNet50","EffNetB0","Lightweight+SE","DualAttention","EfficientNetB4"),
  Threshold = c(res1_ft$threshold, res2_ft$threshold, res3_ft$threshold, res4_ft$threshold, res5_ft$threshold),
  Accuracy  = c(res1_ft$accuracy,   res2_ft$accuracy,   res3_ft$accuracy,   res4_ft$accuracy,   res5_ft$accuracy),
  Precision = c(res1_ft$precision,  res2_ft$precision,  res3_ft$precision,  res4_ft$precision,  res5_ft$precision),
  Recall    = c(res1_ft$recall,     res2_ft$recall,     res3_ft$recall,     res4_ft$recall,     res5_ft$recall),
  F1        = c(res1_ft$F1,         res2_ft$F1,         res3_ft$F1,         res4_ft$F1,         res5_ft$F1),
  AUC       = c(res1_ft$AUC,        res2_ft$AUC,        res3_ft$AUC,        res4_ft$AUC,        res5_ft$AUC)
)
print(summary_ft)
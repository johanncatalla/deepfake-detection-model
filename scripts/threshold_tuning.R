library(pROC)
library(caret)

#––– Helper function: tune & evaluate one model –––#
tune_and_eval <- function(true_labels, pred_probs, positive = 1L) {
  # 1) Build ROC object
  roc_obj <- roc(true_labels, as.numeric(pred_probs))
  
  # 2) Find optimal threshold (Youden's J)
  opt <- coords(
    roc_obj, "best",
    ret = c("threshold","sensitivity","specificity"),
    transpose = FALSE
  )
  # coerce each to a number
  thr  <- as.numeric(opt["threshold"])
  sens <- as.numeric(opt["sensitivity"])
  spec <- as.numeric(opt["specificity"])
  
  cat(
    "Optimal threshold:", round(thr,3),
    "Sens:",        round(sens,3),
    "Spec:",        round(spec,3), "\n"
  )
  
  # 3) Apply threshold
  pred_labels_opt <- ifelse(pred_probs >= thr, positive, 1L - positive)
  
  # 4) Confusion matrix
  cm <- confusionMatrix(
    factor(pred_labels_opt, levels = c(0,1)),
    factor(true_labels,     levels = c(0,1)),
    positive = as.character(positive)
  )
  
  # 5) AUC
  auc_val <- as.numeric(auc(roc_obj))
  
  # 6) Return
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

#––– Run for each model –––#

# Assuming you already have:
#  true_labels1, pred_probs1
#  true_labels1, pred_probs2
#  true_labels1, pred_probs3
#  true_labels1, pred_probs4

res1 <- tune_and_eval(true_labels1, pred_probs1)
res2 <- tune_and_eval(true_labels1, pred_probs2)
res3 <- tune_and_eval(true_labels1, pred_probs3)
res4 <- tune_and_eval(true_labels1, pred_probs4)

#––– Summarize–––#

summary_df <- data.frame(
  Model      = c("ResNet50", "EffNetB0", "Lightweight+SE", "DualAttention"),
  Threshold  = c(res1$threshold, res2$threshold, res3$threshold, res4$threshold),
  Accuracy   = c(res1$accuracy,  res2$accuracy,  res3$accuracy,  res4$accuracy),
  Precision  = c(res1$precision, res2$precision, res3$precision, res4$precision),
  Recall     = c(res1$recall,    res2$recall,    res3$recall,    res4$recall),
  F1         = c(res1$F1,        res2$F1,        res3$F1,        res4$F1),
  AUC        = c(res1$AUC,       res2$AUC,       res3$AUC,       res4$AUC)
)

print(summary_df)

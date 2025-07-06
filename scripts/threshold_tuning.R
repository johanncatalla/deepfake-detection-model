library(pROC)
library(caret)

#––– Helper function: tune & evaluate one model –––#
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

res1 <- tune_and_eval(true_labels1, pred_probs1)
res2 <- tune_and_eval(true_labels1, pred_probs2)
res3 <- tune_and_eval(true_labels1, pred_probs3)
res4 <- tune_and_eval(true_labels1, pred_probs4)
res5 <- tune_and_eval(true_labels1, pred_probs5)  

#––– Summarize–––#

summary_df <- data.frame(
  Model      = c("ResNet50", "EffNetB0", "Lightweight+SE", "DualAttention", "EfficientNetB4"),
  Threshold  = c(res1$threshold, res2$threshold, res3$threshold, res4$threshold, res5$threshold),
  Accuracy   = c(res1$accuracy,  res2$accuracy,  res3$accuracy,  res4$accuracy,  res5$accuracy),
  Precision  = c(res1$precision, res2$precision, res3$precision, res4$precision, res5$precision),
  Recall     = c(res1$recall,    res2$recall,    res3$recall,    res4$recall,    res5$recall),
  F1         = c(res1$F1,        res2$F1,        res3$F1,        res4$F1,        res5$F1),
  AUC        = c(res1$AUC,       res2$AUC,       res3$AUC,       res4$AUC,       res5$AUC)
)

print(summary_df)

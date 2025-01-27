rm(list=ls())

# Download the required libraries if not installed
if (!require("effsize")) install.packages("effsize")
library(effsize)

dsc13_java_full<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-1.3_full_java.csv",header=TRUE)
dsc13_java_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-1.3_qlora_java.csv",header=TRUE)
dsc13_python_full<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-1.3_full_python.csv",header=TRUE)
dsc13_python_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-1.3_qlora_python.csv",header=TRUE)

print("**********************DeepSeek-Coder 1.3B Java [Full vs. QLoRA] *********************************")

#Baseline Comparison#
res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(dsc13_java_full$BLEU,dsc13_java_qlora$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_full$ROUGE_L,dsc13_java_qlora$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_full$BERTScoreF1,dsc13_java_qlora$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_full$METEOR,dsc13_java_qlora$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_full$ChrF,dsc13_java_qlora$ChrF,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_full$SIDE_score,dsc13_java_qlora$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(dsc13_java_full$BLEU,dsc13_java_qlora$BLEU)
cliff.delta(dsc13_java_full$ROUGE_L,dsc13_java_qlora$ROUGE_L)
cliff.delta(dsc13_java_full$BERTScoreF1,dsc13_java_qlora$BERTScoreF1)
cliff.delta(dsc13_java_full$METEOR,dsc13_java_qlora$METEOR)
cliff.delta(dsc13_java_full$ChrF,dsc13_java_qlora$ChrF)
cliff.delta(dsc13_java_full$SIDE_score,dsc13_java_qlora$SIDE_score)
print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)


print("**********************DeepSeek-Coder 1.3B Python [Full vs. QLoRA] *********************************")

res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(dsc13_python_full$BLEU,dsc13_python_qlora$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_full$ROUGE_L,dsc13_python_qlora$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_full$BERTScoreF1,dsc13_python_qlora$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_full$METEOR,dsc13_python_qlora$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_full$ChrF,dsc13_python_qlora$ChrF,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_full$SIDE_score,dsc13_python_qlora$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(dsc13_python_full$BLEU,dsc13_python_qlora$BLEU)
cliff.delta(dsc13_python_full$ROUGE_L,dsc13_python_qlora$ROUGE_L)
cliff.delta(dsc13_python_full$BERTScoreF1,dsc13_python_qlora$BERTScoreF1)
cliff.delta(dsc13_python_full$METEOR,dsc13_python_qlora$METEOR)
cliff.delta(dsc13_python_full$ChrF,dsc13_python_qlora$ChrF)
cliff.delta(dsc13_python_full$SIDE_score,dsc13_python_qlora$SIDE_score)

print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)




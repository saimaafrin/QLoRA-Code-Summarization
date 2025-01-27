rm(list=ls())

# Download the required libraries if not installed
if (!require("effsize")) install.packages("effsize")
library(effsize)

dsc13_java_full<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-1.3_full_java.csv",header=TRUE)
dsc13_java_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-1.3_qlora_java.csv",header=TRUE)
dsc13_python_full<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-1.3_full_python.csv",header=TRUE)
dsc13_python_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-1.3_qlora_python.csv",header=TRUE)

phi3_java_full<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_phi-3.8-full_java.csv",header=TRUE)
phi3_java_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_phi-3_qlora_java.csv",header=TRUE)
phi3_python_full<-read.csv("projects/QLoRA/results/csv_files/metrics/metric_phi-3.8-full_python.csv",header=TRUE)
phi3_python_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_phi-3_qlora_python.csv",header=TRUE)


#Baseline Comparison#
print("********************** Java - Full - [dsc13 vs. phi3] *********************************")
res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(dsc13_java_full$BLEU,phi3_java_full$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_full$ROUGE_L,phi3_java_full$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_full$BERTScoreF1,phi3_java_full$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_full$METEOR,phi3_java_full$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_full$ChrF,phi3_java_full$ChrF,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_full$SIDE_score,phi3_java_full$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(dsc13_java_full$BLEU,phi3_java_full$BLEU)
cliff.delta(dsc13_java_full$ROUGE_L,phi3_java_full$ROUGE_L)
cliff.delta(dsc13_java_full$BERTScoreF1,phi3_java_full$BERTScoreF1)
cliff.delta(dsc13_java_full$METEOR,phi3_java_full$METEOR)
cliff.delta(dsc13_java_full$ChrF,phi3_java_full$ChrF)
cliff.delta(dsc13_java_full$SIDE_score,phi3_java_full$SIDE_score)

print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)

print("------- New Comparasion -------")

print("********************** JAVA - QLoRA - [dsc13 vs. phi3] *********************************")
res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(dsc13_java_qlora$BLEU,phi3_java_qlora$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_qlora$ROUGE_L,phi3_java_qlora$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_qlora$BERTScoreF1,phi3_java_qlora$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_qlora$METEOR,phi3_java_qlora$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_qlora$ChrF,phi3_java_qlora$ChrF,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_java_qlora$SIDE_score,phi3_java_qlora$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(dsc13_java_qlora$BLEU,phi3_java_qlora$BLEU)
cliff.delta(dsc13_java_qlora$ROUGE_L,phi3_java_qlora$ROUGE_L)
cliff.delta(dsc13_java_qlora$BERTScoreF1,phi3_java_qlora$BERTScoreF1)
cliff.delta(dsc13_java_qlora$METEOR,phi3_java_qlora$METEOR)
cliff.delta(dsc13_java_qlora$ChrF,phi3_java_qlora$ChrF)
cliff.delta(dsc13_java_qlora$SIDE_score,phi3_java_qlora$SIDE_score)

print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)

print("------- New Comparasion -------")

print("********************** Python - Full [dsc13 vs. phi3] *********************************")

res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(dsc13_python_full$BLEU,phi3_python_full$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_full$ROUGE_L,phi3_python_full$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_full$BERTScoreF1,phi3_python_full$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_full$METEOR,phi3_python_full$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_full$ChrF,phi3_python_full$ChrF,alternative="two.side",paired=TRUE)$p.value)
#res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_full$SIDE_score,phi3_python_full$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(dsc13_python_full$BLEU,phi3_python_full$BLEU)
cliff.delta(dsc13_python_full$ROUGE_L,phi3_python_full$ROUGE_L)
cliff.delta(dsc13_python_full$BERTScoreF1,phi3_python_full$BERTScoreF1)
cliff.delta(dsc13_python_full$METEOR,phi3_python_full$METEOR)
cliff.delta(dsc13_python_full$ChrF,phi3_python_full$ChrF)
#cliff.delta(dsc13_python_full$ChrF,phi3_python_full$SIDE_score)

print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)


print("------- New Comparasion -------")

print("********************** Python - QLoRA [dsc13 vs. phi3] *********************************")

res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(dsc13_python_qlora$BLEU,phi3_python_qlora$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_qlora$ROUGE_L,phi3_python_qlora$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_qlora$BERTScoreF1,phi3_python_qlora$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_qlora$METEOR,phi3_python_qlora$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_qlora$ChrF,phi3_python_qlora$ChrF,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(dsc13_python_qlora$SIDE_score,phi3_python_qlora$SIDE_score,alternative="two.side",paired=TRUE)$p.value)

cliff.delta(dsc13_python_qlora$BLEU,phi3_python_qlora$BLEU)
cliff.delta(dsc13_python_qlora$ROUGE_L,phi3_python_qlora$ROUGE_L)
cliff.delta(dsc13_python_qlora$BERTScoreF1,phi3_python_qlora$BERTScoreF1)
cliff.delta(dsc13_python_qlora$METEOR,phi3_python_qlora$METEOR)
cliff.delta(dsc13_python_qlora$ChrF,phi3_python_qlora$ChrF)
cliff.delta(dsc13_python_qlora$ChrF,phi3_python_qlora$SIDE_score)

print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)




rm(list=ls())

# Download the required libraries if not installed
if (!require("effsize")) install.packages("effsize")
library(effsize)

DSC67_java_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-6.7_qlora_java.csv",header=TRUE)
DSC33_java_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC33_qlora_java.csv",header=TRUE)
DSC67_python_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metric_DSC-6.7-qlora_python.csv",header=TRUE)
DSC33_python_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC33_qlora_python.csv",header=TRUE)

print("********************** DSC 1.3B vs DSC 6.7B [Java QLoRA] *********************************")

#Baseline Comparison#
res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(DSC67_java_qlora$BLEU,DSC33_java_qlora$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC67_java_qlora$ROUGE_L,DSC33_java_qlora$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC67_java_qlora$BERTScoreF1,DSC33_java_qlora$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC67_java_qlora$METEOR,DSC33_java_qlora$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC67_java_qlora$ChrF,DSC33_java_qlora$ChrF,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC67_java_qlora$SIDE_score,DSC33_java_qlora$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(DSC67_java_qlora$BLEU,DSC33_java_qlora$BLEU)
cliff.delta(DSC67_java_qlora$ROUGE_L,DSC33_java_qlora$ROUGE_L)
cliff.delta(DSC67_java_qlora$BERTScoreF1,DSC33_java_qlora$BERTScoreF1)
cliff.delta(DSC67_java_qlora$METEOR,DSC33_java_qlora$METEOR)
cliff.delta(DSC67_java_qlora$ChrF,DSC33_java_qlora$ChrF)
cliff.delta(DSC67_java_qlora$SIDE_score,DSC33_java_qlora$SIDE_score)
print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)


print("********************** DSC 1.3B vs DSC 6.7B [Python QLoRA] *********************************")

res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(DSC67_python_qlora$BLEU,DSC33_python_qlora$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC67_python_qlora$ROUGE_L,DSC33_python_qlora$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC67_python_qlora$BERTScoreF1,DSC33_python_qlora$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC67_python_qlora$METEOR,DSC33_python_qlora$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC67_python_qlora$ChrF,DSC33_python_qlora$ChrF,alternative="two.side",paired=TRUE)$p.value)
#res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC67_python_qlora$SIDE_score,DSC33_python_qlora$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(DSC67_python_qlora$BLEU,DSC33_python_qlora$BLEU)
cliff.delta(DSC67_python_qlora$ROUGE_L,DSC33_python_qlora$ROUGE_L)
cliff.delta(DSC67_python_qlora$BERTScoreF1,DSC33_python_qlora$BERTScoreF1)
cliff.delta(DSC67_python_qlora$METEOR,DSC33_python_qlora$METEOR)
cliff.delta(DSC67_python_qlora$ChrF,DSC33_python_qlora$ChrF)
#cliff.delta(DSC67_python_qlora$SIDE_score,DSC33_python_qlora$SIDE_score)

print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)




rm(list=ls())

# Download the required libraries if not installed
if (!require("effsize")) install.packages("effsize")
library(effsize)

DSC13_java_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-1.3_qlora_java.csv",header=TRUE)
DSC67_java_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_DSC-6.7_qlora_java.csv",header=TRUE)
DSC13_python_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metric_DSC-1.3-qlora_python.csv",header=TRUE)
DSC67_python_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metric_DSC-6.7-qlora_python.csv",header=TRUE)

print("********************** DSC 1.3B vs DSC 6.7B [Java QLoRA] *********************************")

#Baseline Comparison#
res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(DSC13_java_qlora$BLEU,DSC67_java_qlora$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC13_java_qlora$ROUGE_L,DSC67_java_qlora$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC13_java_qlora$BERTScoreF1,DSC67_java_qlora$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC13_java_qlora$METEOR,DSC67_java_qlora$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC13_java_qlora$ChrF,DSC67_java_qlora$ChrF,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC13_java_qlora$SIDE_score,DSC67_java_qlora$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(DSC13_java_qlora$BLEU,DSC67_java_qlora$BLEU)
cliff.delta(DSC13_java_qlora$ROUGE_L,DSC67_java_qlora$ROUGE_L)
cliff.delta(DSC13_java_qlora$BERTScoreF1,DSC67_java_qlora$BERTScoreF1)
cliff.delta(DSC13_java_qlora$METEOR,DSC67_java_qlora$METEOR)
cliff.delta(DSC13_java_qlora$ChrF,DSC67_java_qlora$ChrF)
cliff.delta(DSC13_java_qlora$SIDE_score,DSC67_java_qlora$SIDE_score)
print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)


print("********************** DSC 1.3B vs DSC 6.7B [Python QLoRA] *********************************")

res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(DSC13_python_qlora$BLEU,DSC67_python_qlora$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC13_python_qlora$ROUGE_L,DSC67_python_qlora$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC13_python_qlora$BERTScoreF1,DSC67_python_qlora$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC13_python_qlora$METEOR,DSC67_python_qlora$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC13_python_qlora$ChrF,DSC67_python_qlora$ChrF,alternative="two.side",paired=TRUE)$p.value)
#res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(DSC13_python_qlora$SIDE_score,DSC67_python_qlora$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(DSC13_python_qlora$BLEU,DSC67_python_qlora$BLEU)
cliff.delta(DSC13_python_qlora$ROUGE_L,DSC67_python_qlora$ROUGE_L)
cliff.delta(DSC13_python_qlora$BERTScoreF1,DSC67_python_qlora$BERTScoreF1)
cliff.delta(DSC13_python_qlora$METEOR,DSC67_python_qlora$METEOR)
cliff.delta(DSC13_python_qlora$ChrF,DSC67_python_qlora$ChrF)
#cliff.delta(DSC13_python_qlora$SIDE_score,DSC67_python_qlora$SIDE_score)

print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)




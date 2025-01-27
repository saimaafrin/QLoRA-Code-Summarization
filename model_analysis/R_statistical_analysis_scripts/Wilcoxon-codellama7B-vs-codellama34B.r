rm(list=ls())

# Download the required libraries if not installed
if (!require("effsize")) install.packages("effsize")
library(effsize)

CL7_java_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_CL-7_qlora_java.csv",header=TRUE)
CL34_java_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metrics_for_R_scripts/metricR_CL-34_qlora_java.csv",header=TRUE)
CL7_python_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metric_CL-7-qlora_python.csv",header=TRUE)
CL34_python_qlora<-read.csv("projects/QLoRA/results/csv_files/metrics/metric_CL-34-qlora_python.csv",header=TRUE)


CL34_java_qlora <- CL34_java_qlora[1:10951, ]
CL34_python_qlora <- CL34_python_qlora[1:10951, ]

print("********************** CodeLlama 7B vs CodeLlama 34B [Java QLoRA] *********************************")

#Baseline Comparison#
res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(CL7_java_qlora$BLEU,CL34_java_qlora$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(CL7_java_qlora$ROUGE_L,CL34_java_qlora$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(CL7_java_qlora$BERTScoreF1,CL34_java_qlora$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(CL7_java_qlora$METEOR,CL34_java_qlora$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(CL7_java_qlora$ChrF,CL34_java_qlora$ChrF,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(CL7_java_qlora$SIDE_score,CL34_java_qlora$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(CL7_java_qlora$BLEU,CL34_java_qlora$BLEU)
cliff.delta(CL7_java_qlora$ROUGE_L,CL34_java_qlora$ROUGE_L)
cliff.delta(CL7_java_qlora$BERTScoreF1,CL34_java_qlora$BERTScoreF1)
cliff.delta(CL7_java_qlora$METEOR,CL34_java_qlora$METEOR)
cliff.delta(CL7_java_qlora$ChrF,CL34_java_qlora$ChrF)
cliff.delta(CL7_java_qlora$SIDE_score,CL34_java_qlora$SIDE_score)
print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)


print("********************** CodeLlama 7B vs CodeLlama 34B [Python QLoRA] *********************************")
#print(length(CL7_python_qlora$BLEU))
#print(length(CL34_python_qlora$BLEU))
CL7_python_qlora <- CL7_python_qlora[1:10951, ]


res=list(Wilcoxon.p=c())

res$Wilcoxon.p=(wilcox.test(CL7_python_qlora$BLEU,CL34_python_qlora$BLEU,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(CL7_python_qlora$ROUGE_L,CL34_python_qlora$ROUGE_L,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(CL7_python_qlora$BERTScoreF1,CL34_python_qlora$BERTScoreF1,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(CL7_python_qlora$METEOR,CL34_python_qlora$METEOR,alternative="two.side",paired=TRUE)$p.value)
res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(CL7_python_qlora$ChrF,CL34_python_qlora$ChrF,alternative="two.side",paired=TRUE)$p.value)
#res$Wilcoxon.p=append(res$Wilcoxon.p, wilcox.test(CL7_python_qlora$SIDE_score,CL34_python_qlora$SIDE_score,alternative="two.side",paired=TRUE)$p.value)


cliff.delta(CL7_python_qlora$BLEU,CL34_python_qlora$BLEU)
cliff.delta(CL7_python_qlora$ROUGE_L,CL34_python_qlora$ROUGE_L)
cliff.delta(CL7_python_qlora$BERTScoreF1,CL34_python_qlora$BERTScoreF1)
cliff.delta(CL7_python_qlora$METEOR,CL34_python_qlora$METEOR)
cliff.delta(CL7_python_qlora$ChrF,CL34_python_qlora$ChrF)
#cliff.delta(CL7_python_qlora$SIDE_score,CL34_python_qlora$SIDE_score)

print("**************************************************************************************************")

res=data.frame(res)
res$Wilcoxon.p=p.adjust(res$Wilcoxon.p, method="holm")
print(res)




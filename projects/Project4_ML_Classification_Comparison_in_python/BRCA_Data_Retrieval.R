library("TCGA2STAT")
library("TCGAbiolinks")
options(timeout=10000)
rnaseq.RPKM.BRCA <- getTCGA(disease="BRCA", data.type="RNASeq2", type="RPKM",clinical = TRUE)
rnaseq.count.BRCA <- getTCGA(disease="BRCA", data.type="RNASeq2", type="count",clinical = TRUE)
query_BRCA <- GDCquery(project = "TCGA-BRCA",
data.category = "Clinical",
data.type = "Clinical Supplement",
data.format = "BCR Biotab")
GDCdownload(query_BRCA)
clinical.BCRtab.all_BRCA <- GDCprepare(query_BRCA)
BRCA.subtype <- TCGAquery_subtype(tumor = "brca")
BRCA_Subtypes<-data.frame(Patients=brca.subtype$patient,Subtypes=brca.subtype$BRCA_Subtype_PAM50)
BRCA_Subtypes<-BRCA_Subtypes[BRCA_Subtypes$Subtypes!="NA",]
BRCA__Expression<-rnaseq.RPKM.BRCA$dat
PAM50_Genes<-read.table("Project2/PAM50_Genes.txt",header = FALSE,col.names = "Genes")
PAM50_Expression<-BRCA__Expression[PAM50_Genes$Genes,]
PAM50_Expression_unique<-PAM50_Expression[,!duplicated(gsub("-01A-.*|-01B-.*|-11A-.*|-11B-.*|-06A-.*|-06B-.*","",colnames(PAM50_Expression)))]
colnames(PAM50_Expression_unique)<-gsub("-01A-.*|-01B-.*|-11A-.*|-11B-.*|-06A-.*|-06B-.*","",colnames(PAM50_Expression_unique))
Overlap_Samples<-intersect(BRCA_Subtypes$Patients,colnames(PAM50_Expression_unique))
PAM50_Expression_clean<-PAM50_Expression_unique[,match(Overlap_Samples,colnames(PAM50_Expression_unique))]
PAM50_Subtype_clean<-BRCA_Subtypes[match(Overlap_Samples,BRCA_Subtypes$Patients),]
View(PAM50_Expression_clean)
write.table(PAM50_Expression_clean,file = "Project2/BRCA_PAM50_Expression.txt",sep = ",",col.names = TRUE,row.names = TRUE,quote = FALSE)
write.table(PAM50_Subtype_clean,file = "Project2/BRCA_Subtypes.txt",sep = ",",col.names = TRUE,row.names = TRUE,quote = FALSE)


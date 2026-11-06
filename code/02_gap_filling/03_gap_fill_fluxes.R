

rm(list=ls(all=TRUE)) 
library(Rcpp)
library(REddyProc)




#data <-read.csv(file="C:/ammara_MD/a_harvard/Project/data/Evapotranspiration/gaps_EMS.csv",header=TRUE)
#data <-read.csv(file="C:/ammara_MD/a_harvard/Project/data/Evapotranspiration/gaps_HEM.csv",header=TRUE)
#data <-read.csv(file="C:/ammara_MD/a_harvard/Project/HF_paper/gaps_EMS.csv",header=TRUE)

data <-read.csv(file="C:/ammara_MD/a_harvard/Project/HF_paper/gaps_HEM.csv",header=TRUE)

oct<-(data)

oct=oct[,-1]

EddyDataWithPosix.F <- fConvertTimeToPosix(oct, 'YDH', Year.s = 'Year', Day.s = 'DoY', Hour.s = 'Hour')


#+++ Processing with ustar filtering before 

# hourly  

EddyProc.C<- sEddyProc$new('oct', EddyDataWithPosix.F, c('NEE', 'GPP','LE','H','Rg','Tair','VPD','Ustar','PAR','RH','PA','NETRAD','WS'),DTS=24)

# half hour
EddyProc.C<- sEddyProc$new('oct', EddyDataWithPosix.F, c('NEE', 'LE','H','Rg','Tair','VPD','Ustar','PAR','RH','PA','NETRAD','WS'),DTS=48)
################################################################################


uStarTh <- EddyProc.C$sEstUstarThreshold()$uStarTh

EddyProc.C$sMDSGapFillAfterUstar('NEE')
EddyProc.C$sExportResults()

#EddyProc.C$sMDSGapFillAfterUstar('GPP')
#EddyProc.C$sExportResults()


EddyProc.C$sMDSGapFillAfterUstar('LE')
EddyProc.C$sExportResults()

EddyProc.C$sMDSGapFillAfterUstar('H')
EddyProc.C$sExportResults()

EddyProc.C$sMDSGapFillAfterUstar('Rg')
EddyProc.C$sExportResults()

EddyProc.C$sMDSGapFillAfterUstar('Tair')
EddyProc.C$sExportResults()

EddyProc.C$sMDSGapFillAfterUstar('VPD')
EddyProc.C$sExportResults()

# done 
EddyProc.C$sMDSGapFillAfterUstar('PAR')
EddyProc.C$sExportResults()

EddyProc.C$sMDSGapFillAfterUstar('RH')
EddyProc.C$sExportResults()

EddyProc.C$sMDSGapFillAfterUstar('PA')
EddyProc.C$sExportResults()


EddyProc.C$sMDSGapFillAfterUstar('NETRAD')
EddyProc.C$sExportResults()

EddyProc.C$sMDSGapFillAfterUstar('WS')
EddyProc.C$sExportResults()

#colnames(EddyProc.C$sExportResults()) # Note the collumns with suffix _WithUstar	

filled1<- EddyProc.C$sExportResults()

filled2=data.frame(cbind(Year=EddyDataWithPosix.F$Year,DoY=EddyDataWithPosix.F$DoY,Hour=EddyDataWithPosix.F$Hour,
LE=filled1$LE_uStar_f))

#PA=filled1$PA_uStar_f,NETRAD=filled1$NETRAD_uStar_f,WS=filled1$WS_uStar_f
write.csv(filled2,"C:/ammara_MD/a_harvard/Project/HF_paper/filled_HEM2.csv")


#write.csv(filled2,"C:/ammara_MD/a_harvard/Project/HF_paper/filled_EMS2.csv")
filled=data.frame(cbind(Year=EddyDataWithPosix.F$Year,DoY=EddyDataWithPosix.F$DoY,Hour=EddyDataWithPosix.F$Hour,
NEE=filled$NEE_uStar_f,LE=filled$LE_uStar_f,H=filled$H_uStar_f,Rg=filled$Rg_uStar_f,
Tair=filled$Tair_uStar_f,VPD=filled$VPD_uStar_f,PAR=filled$PAR_uStar_f,RH=filled$RH_uStar_f,
PA=filled$PA_uStar_f,NETRAD=filled$NETRAD_uStar_f,WS=filled$WS_uStar_f))
                        
write.csv(filled,"C:/ammara_MD/a_harvard/Project/HF_paper/filled_HEM.csv")

#filled1=data.frame(cbind(Year=EddyDataWithPosix.F$Year,DoY=EddyDataWithPosix.F$DoY,Hour=EddyDataWithPosix.F$Hour,NETRAD=filled1$NETRAD_uStar_f,WS=filled1$WS_uStar_f))

write.csv(filled,"C:/ammara_MD/a_harvard/Project/HF_paper/filled_HEM.csv")

#filled=data.frame(cbind(Year=EddyDataWithPosix.F$Year,DoY=EddyDataWithPosix.F$DoY,Hour=EddyDataWithPosix.F$Hour,NEE=filled$NEE_uStar_f, GPP=filled$GPP_uStar_f,LE=filled$LE_uStar_f,H=filled$H_uStar_f,Rg=filled$Rg_uStar_f,Tair=filled$Tair_uStar_f,VPD=filled$VPD_uStar_f,PAR=filled$PAR_uStar_f))

#write.csv(filled,"C:/ammara_MD/a_harvard/Project/data/Evapotranspiration/filled_HEM.csv")








#write.csv(filled,"C:/ammara_MD/a_harvard/Project/data/Evapotranspiration/filled_HEM.csv")



#write.csv(filled,"C:/ammara_MD/flux_tower_data/Tower_WISP/correct_flux/all_data_2021/potatoes_2021/filled.csv")

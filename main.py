
from os import stat
from pandas.core.reshape.merge import merge
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly as plt
from streamlit.proto.DataFrame_pb2 import DataFrame
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from fpdf import FPDF 
import plotly.graph_objects as go
from pandas.plotting import table 
import  chart_studio as py


st.set_page_config(layout="wide")
# title of the app
st.title("Vacuum tester Visualization")
# Add a sidebar
st.sidebar.subheader("Setting")
uploaded_files = st.sidebar.file_uploader(label='Upload CSV File or Excel file', type=['csv', 'xlsx'],
                                          accept_multiple_files=True)
Merge = st.sidebar.button("Merge")
step = st.sidebar.number_input('step', value=12)
totQ = st.sidebar.number_input('Tot', value=912)
Defect = st.selectbox(label="Electrical Parameter", options=('ALL',"nur Schlecht","nur Gut","RS",  "FL", "C1", "DF", 'Q',"FS"))

global uploaded_file
i=0
for uploaded_file in uploaded_files:
    i=i+1
    data = pd.read_csv(uploaded_file, skiprows=10, sep=";", header=0, names=("Nbr", "RS", "FS", "FL", "C1", "DF", "Q", "Remarks"), index_col="Nbr")                
    data = data.fillna('') 
    data = data.reindex(columns=['Quartz', 'RS', 'C1','FL','FS','DF','Q','Remarks'])
    #creer une colonne quarz
    totW=round(len(data)/(round(totQ/step)))
    st.sidebar.write(uploaded_file.name,'Number of Wafer',totW)
    dW=pd.DataFrame(index=np.arange(totQ), columns=np.arange(1))
    dp=pd.DataFrame()
    p=0
    for t in range(0,totQ):
        dW.loc[t,:]=t+1
    for c in range (0,totW): 
        dp=dp.append(dW)
    for j in range(0,len(dp),step):
        data.iloc[p,0]=dp.iloc[j,0] 
        p=p+1
    expander_1 = st.beta_expander(uploaded_file.name)
    # Calcul Resistance
    R = data['RS']
    R_all = R.loc[R != 0]
    mR=R_all.mean().round(3)
    sR=R_all.std().round(3)
    # Cacul frequence
    F = data['FL']
    F_all=F.loc[F != 0]
    mF=F_all.mean().round(4)
    sF=(1000*F_all.std()).round(2)
    # Calcul C1
    C = data['C1']
    C_All = C.loc[C != 0]
    mC=C_All.mean().round(3)
    sC=C_All.std().round(3)
    #statistique
    Kurz=len(data[data['Remarks'].str.contains('Short circuit')])
    NoF=len(data[data['Remarks'].str.contains('no Fs')])
    Leer=len(data[data['Remarks'].str.contains('Empty')])
    infC1=len(data.loc[data['Remarks'] == '<C1'])
    supRs=len(data.loc[data['Remarks'] == '>Rs'])
    RsC1=len(data.loc[data['Remarks'] == '>Rs <C1'])
    infFl=len(data[data['Remarks'].str.contains('<DF')])
    supFl=len(data[data['Remarks'].str.contains('>DF')])
    Gut=len(data.loc[data['Remarks'] == ""])
    schlecht=len(data.loc[data['Remarks'] !=''])
    Tot=len(data)
    with expander_1:
        coll1,coll2 = st.beta_columns((3,1))    
        if Defect == "ALL":
            with coll1:
                df = data
                st.write(data)
            with coll2:
                liste1= ["kurz","No F","Leer","<C1",">Rs",">Rs & <C1","<Df",">Df","Schlecht","Gut","Tot"]
                liste2=[Kurz,NoF,Leer,infC1,supRs,RsC1,infFl,supFl,schlecht,Gut,Tot]
                statik=pd.DataFrame({'#Q':liste2},index=liste1)
                st.table(statik)
            with coll1:
                l1=["kurz","No F","Leer","<C1",">Rs",">Rs & <C1","<Df",">Df","Gut"]
                l2=[Kurz,NoF,Leer,infC1,supRs,RsC1,infFl,supFl,Gut]
                fig2=px.pie(values=l2,names=l1,title='Statistik')
                st.write(fig2)
                param = {'Rs[kOhm]':{'µ':mR,'σ':sR},'FL[Hz]':{'µ':mF,'σ':sF},'C1[fF]':{'µ':mC,'σ':sC}}
                st.table(param)
                #creer histogramme et les génerer
                figR= px.histogram(R_all,x='RS')
                st.plotly_chart(figR)  
                figC= px.histogram(C_All,x='C1')
                st.plotly_chart(figC) 
                figF= px.histogram(F_all,x='FL')
                st.plotly_chart(figF)  
        if Defect == "nur Schlecht":
            with coll1:
                df=data.loc[data['Remarks'] !='']
                st.write(df)
            with coll2:
                liste1= ["kurz","No F","Leer","<C1",">Rs",">Rs & <C1","<Df",">Df","Schlecht","Gut","Tot"]
                liste2=[Kurz,NoF,Leer,infC1,supRs,RsC1,infFl,supFl,schlecht,Gut,Tot]
                statik=pd.DataFrame({'#Q':liste2},index=liste1)
                st.table(statik)
            with coll1:
                l1=["kurz","No F","Leer","<C1",">Rs",">Rs & <C1","<Df",">Df"]
                l2=[Kurz,NoF,Leer,infC1,supRs,RsC1,infFl,supFl]
                fig2=px.pie(values=l2,names=l1,title='Statistik')
                st.write(fig2)
        if Defect == "nur Gut":
            with coll1:
                df=data.loc[data['Remarks'] == ""]
                st.write(df)
                # Calcul Resistance
                R = df['RS']
                mR=R.mean().round(1)
                sR=R.std().round(0)
                # Cacul frequence
                F = df['FL']
                mF=F.mean().round(3)
                sF=1000*F.std().round(3)
                # Calcul C1
                C = df['C1']
                mC=C.mean().round(2)
                sC=C.std().round(2)
                st.write('Analyze')
                param = {'Rs[kOhm]':{'µ':mR,'σ':sR},'FL[Hz]':{'µ':mF,'σ':sF},'C1[fF]':{'µ':mC,'σ':sC}}
                st.table(param)
        col1,col2 = st.beta_columns((1,1))
        if Defect == "RS":
            RS = data['RS']
            RS = RS.loc[RS != 0]
            df = pd.DataFrame(RS)
            with col1:
                st.dataframe(df.style.highlight_max(axis=0,color='pink'))
                st.write('#### Rs=' ,RS.mean().round(0),'kOhm') 
                st.write('#### std= ',RS.std().round(0),'kOhm') 
            with col2:
                fig1= px.histogram(df,x='RS')
                st.plotly_chart(fig1)  
        if Defect == "FS":
            FS = data['FS']
            FS = FS.loc[FS != 0]
            df = pd.DataFrame(FS)
            with col1:
                st.dataframe(df.style.highlight_max(axis=0,color='pink'))
                st.write('#### FS=', FS.mean().round(3),['kHz'])
                st.write('#### std=', FS.std().round(0),'[kHz]')
            with col2:
                fig1= px.histogram(df,x='FS')
                st.plotly_chart(fig1)  
        if Defect == "FL":
            FL = data['FL']
            FL = FL.loc[FL != 0]
            df = pd.DataFrame(FL)
            with col1:
                st.dataframe(df.style.highlight_max(axis=0,color='pink'))
                st.write('#### FL=', FL.mean().round(3),'[kHz]')
                st.write('#### std=', 1000*(FL.std()).round(3),'[Hz]')
            with col2:
                fig1= px.histogram(df,x='FL')
                st.plotly_chart(fig1)  
        if Defect == "C1":
            C1 = data['C1']
            C1 = C1.loc[C1 != 0]
            df = pd.DataFrame(C1)
            with col1:
                st.dataframe(df.style.highlight_min(axis=0,color='blue'))
                st.write('#### C1=', C1.mean().round(1),'[fF]')
                st.write('#### std[fF]=', C1.std().round(1),'[fF]') 
            with col2:
                 fig1= px.histogram(df,x='C1')
                 st.plotly_chart(fig1) 
        if Defect == "DF":
            DeltaF = pd.to_numeric(data['DF'])
            DeltaF = DeltaF.fillna(9999)
            DeltaF = DeltaF.loc[DeltaF <= 0]
            df = pd.DataFrame(DeltaF)
            with col1:
                st.dataframe(df.style.highlight_min(axis=0,color='yellow'))
                st.write('DF=', DeltaF.mean().round(0),'[Hz]')
                st.write('std=', DeltaF.std().round(0),'[Hz]')   
            with col2:
                fig1= px.histogram(df,x='DF')
                st.plotly_chart(fig1)  
        if Defect == "Q":
            Q = data['Q']
            df = Q.loc[Q != 0]
            with col1:
                st.write(df)
                st.write('#### Q=', df.mean().round(0))
                st.write('#### std=', df.std().round(0))   
            with col2:
                fig1= px.histogram(df,x='Q')
                st.plotly_chart(fig1) 
    if i==1:
        merge_data=pd.DataFrame(data)
    if i >1:
        merge_data=merge_data.append(data)
if Merge:   
        # Calcul Resistance
        R = merge_data['RS']
        R_all = R.loc[R != 0]
        mR=R_all.mean().round(0)
        sR=R_all.std().round(0)
        # Cacul frequence
        F = data['FL']
        F_all=F.loc[F != 0]
        mF=F_all.mean().round(3)
        sF=1000*F_all.std().round(3)
        # Calcul C1
        C = data['C1']
        C_All = C.loc[C != 0]
        mC=C_All.mean().round(1)
        sC=C_All.std().round(1)
        #statistique
        Kurz=len(merge_data[merge_data['Remarks'].str.contains('Short circuit')])
        NoF=len(merge_data[merge_data['Remarks'].str.contains('no Fs')])
        Leer=len(merge_data[merge_data['Remarks'].str.contains('Empty')])
        infC1=len(merge_data.loc[merge_data['Remarks'] == '<C1'])
        supRs=len(merge_data.loc[merge_data['Remarks'] == '>Rs'])
        RsC1=len(merge_data.loc[merge_data['Remarks'] == '>Rs <C1'])
        infFl=len(merge_data[merge_data['Remarks'].str.contains('<DF')])
        supFl=len(merge_data[merge_data['Remarks'].str.contains('>DF')])
        Gut=len(merge_data.loc[merge_data['Remarks'] == ""])
        schlecht=len(merge_data.loc[merge_data['Remarks'] !=''])
        Tot=len(merge_data)
        T1,T2 = st.beta_columns((3,2)) 
        if Defect == "ALL":
            with T1:
                st.write('Merged File')
                st.dataframe(merge_data) 
                list3=['µ','σ']
                parameter=pd.DataFrame({"Rs[kOhm]":[mR,sR],"C1[fF]":[mC,sC],"FL[Hz]":[mF,sF]},index=list3)
                param_All = {'Rs[kOhm]':{'µ':mR,'σ':sR},'FL[Hz]':{'µ':mF,'σ':sF},'C1[fF]':{'µ':mC,'σ':sC}}
                st.table(param_All)
                ax = plt.subplot(122,frame_on=False) # no visible frame
                ax.xaxis.set_visible(False)  # hide the x axis
                ax.yaxis.set_visible(False)  # hide the y axis
                table(ax, parameter,rowLabels=['']*parameter.shape[0], loc='top')  # where parameter is your data frame
            with T2:
                liste1= ["kurz","No F","Leer","<C1",">Rs",">Rs & <C1","<Df",">Df","Schlecht","Gut","Tot"]
                liste2=[Kurz,NoF,Leer,infC1,supRs,RsC1,infFl,supFl,schlecht,Gut,Tot]
                statik=pd.DataFrame({'#Q':liste2},index=liste1)
                st.table(statik)
                ax = plt.subplot(122,frame_on=False) # no visible frame
                ax.xaxis.set_visible(False)  # hide the x axis
                ax.yaxis.set_visible(False)  # hide the y axis
                table(ax, statik,rowLabels=['']*statik.shape[0], loc='center')  # where statik is your data frame
                plt.savefig('Stat.png')
            with T1:
                l1=["kurz","No F","Leer","<C1",">Rs",">Rs & <C1","<Df",">Df","Gut"]
                l2=[Kurz,NoF,Leer,infC1,supRs,RsC1,infFl,supFl,Gut]
                fig2=px.pie(values=l2,names=l1,title='Statistik')
                st.write(fig2)  
                fig = plt.figure()
                ax = fig.add_axes([0,0,1,1])
                ax.axis('equal')
                ax.pie(l2, labels =l1,autopct='%1.2f%%')
                plt.savefig('pie.png',loc='center')

            class PDF(FPDF):
                def header(self):
                    # Logo
                    #self.image('logo.png', 10, 10, 35)
                    # font
                    self.set_font('helvetica', 'B', 20)
                    # Padding
                    self.cell(80)
                    # Title
                    self.cell(30, 10, 'Report', border=True, ln=1, align='C')
                    # Line break
                    self.ln(20)
                # Page footer
                def footer(self):
                    # Set position of the footer
                    self.set_y(-15)
                    # set font
                    self.set_font('helvetica', 'I', 8)
                    # Page number
                    self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C') 
            # Create a PDF object
            pdf = PDF('P', 'mm', 'Letter')
            # get total page numbers
            pdf.alias_nb_pages()
            # Set auto page break
            pdf.set_auto_page_break(auto = True, margin = 15)
            #Add Page
            pdf.add_page()
            # specify font
            pdf.image('Stat.png',x=50,y=40,w=100,h=120)
            pdf.image('pie.png',x=60,y=170,w=100,h=90)

            
            pdf.output('Report.pdf')
        if Defect == "nur Schlecht":
            with T1:
                df=merge_data.loc[merge_data['Remarks'] !='']
                st.write(df)
            with T2:
                liste1= ["kurz","No F","Leer","<C1",">Rs",">Rs & <C1","<Df",">Df","Schlecht","Gut","Tot"]
                liste2=[Kurz,NoF,Leer,infC1,supRs,RsC1,infFl,supFl,schlecht,Gut,Tot]
                statik=pd.DataFrame({'#Q':liste2},index=liste1)
                st.table(statik)
            with T1:
                l1=["kurz","No F","Leer","<C1",">Rs",">Rs & <C1","<Df",">Df"]
                l2=[Kurz,NoF,Leer,infC1,supRs,RsC1,infFl,supFl]
                fig2=px.pie(values=l2,names=l1,title='Statistik')
                st.write(fig2)
        if Defect == "nur Gut":
            with T1:
                df=merge_data.loc[merge_data['Remarks'] == ""]
                st.write(df)
                # Calcul Resistance
                R = df['RS']
                mR=R.mean().round(1)
                sR=R.std().round(0)
                # Cacul frequence
                F = df['FL']
                mF=F.mean().round(3)
                sF=1000*F.std().round(3)
                # Calcul C1
                C = df['C1']
                mC=C.mean().round(2)
                sC=C.std().round(2)
                st.write('Analyze')
                param = {'Rs[kOhm]':{'µ':mR,'σ':sR},'FL[Hz]':{'µ':mF,'σ':sF},'C1[fF]':{'µ':mC,'σ':sC}}
                st.table(param)
  
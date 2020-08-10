import streamlit as st
import os
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import re
import altair as alt
import glob

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.layout = html.Div([
    
    html.H1(children='IGI material control',
           style={
               'textAlign':'center',
               #'color': colors['text']
           }),
     html.Div(children='A web application framework', style={
        'textAlign': 'center',
        #'color': colors['text']
    }),
    #input feature num
    html.P("Please give number of dimension need to be measured"),
    dcc.Input(
            id="input_features", type="number",
            debounce=True, value="5",
        ),
    html.P("Please input start date and end date, leave it blank for full range"),
    dcc.Input(id='start_date', value='', type='text'),
    dcc.Input(id='end_date', value='', type='text'),
    dcc.Graph(id='graph-plotly')
])

@app.callback(
    Output('graph-plotly', 'figure'),
    [Input('input_features', 'value'),
     Input('start_date', 'value'),
     Input('end_date', 'value')
    ])
def chart_show(input_value,start_date,end_date):
    feature_num=int(input_value)

    sample_num=32  # Max AQL value for number of sample inspection (tighten)

    path='//Vn01w2k16v18/data/Copyroom/Test_software/Data/Membrane 3000S/'
    path3='//Vn01w2k16v18/data/Copyroom/Test_software/Data/Save/'

    #print(path)
    all_files1=glob.glob(path + '*.xlsx')
    all_files2=glob.glob(path + '*.xlsm')
    all_files=all_files1+all_files2
    #sort file in directory by reverse:
    #all_files = sorted(all_files, reverse = False)
    all_files = sorted(all_files, key=lambda t: os.stat(t).st_mtime)
    #print('number of files: ',len(all_files))
    st.text('number of files: '+str(len(all_files)))
    ### Initial number of feature and sample:

    ### add all dataframe to list of dataframe:


    def create_dataframe(all_files):
        finaldf=pd.DataFrame()
        for filename in all_files:
            try:
                df = pd.read_excel(filename, index_col = None, header = 0)
            except:
                st.text('File cannot be read: '+filename)
                continue
            #Create data frame feature values(newdf)
            newdf=pd.DataFrame()
            for i in range(feature_num+1)[1:]: # add all value to dataframe (depend on num of feature columns)
              df_feature= df['Unnamed: '+str(i)][19:(19+sample_num)] #must be = 19 + sample_num /24 la mac dinh
              #O day thuc hien 2 lenh: 1- xac dinh vi tri cot theo feature, 2- copy row co values tu index 19 tro di
              df_feature=df_feature.reset_index(drop=True)
              #remove not float row and na in df_feature: (not good for some data)
              non_numeric_column=pd.DataFrame(df_feature)[~pd.DataFrame(df_feature).applymap(np.isreal).all(1)]
              num_row_nonnumeric=len(non_numeric_column)
              list_to_exclude=non_numeric_column.index.values.tolist()
              df_feature=df_feature[~df_feature.index.isin(list_to_exclude)]
              #sample_num=len(df_feature)

              newdf=pd.concat([newdf,df_feature],axis=1)
            #print(newdf)  

            #Date transform:
            a=df[['Unnamed: 7']][3:4] #object type - normal is [2:3]
            if a.isna().values:
              a=df[['Unnamed: 7']][2:3]
              date = a.values[0][0] #timestamp type
            else:
              date = a.values[0][0] #timestamp type
            if isinstance(date, str):
              continue
            #Create series date column (row = number of value to record, default = 5)
            date=pd.Series(date)
            datedf=pd.DataFrame()
            for i in range(sample_num):
              datedf=pd.concat([datedf,date],axis=0)

            datedf.reset_index(drop=True, inplace=True)
            newdf.reset_index(drop=True, inplace=True)
            df_combine=pd.concat([datedf,newdf],axis=1) # combine date and features

            #combine all df each filename:

            finaldf=pd.concat([finaldf,df_combine],axis=0)

        #print(finaldf.isnull().sum())
        finaldf.reset_index(drop=True,inplace=True)
        finaldf.sort_values(by=finaldf.columns[0])
        #finaldf

        #finaldf.to_csv('finaldfMMD.csv')

        finaldf.dropna(inplace=True)
        #finaldf

        finaldf.reset_index(drop=True,inplace=True)
        column_name=[]
        column_name.append('Datef')
        for i in range(feature_num+1)[1:]:
          #print(df['Unnamed: '+str(i)][7])
          namecolumn_temp=df['Unnamed: '+str(i)][7]
          namecolumn_temp=namecolumn_temp.replace('.','')
          column_name.append(namecolumn_temp)
        finaldf.columns=column_name
        finaldf.sort_values(by='Datef') #sort dated , not depend on file sorting
        for name in finaldf.columns[1:]:
          try:
            finaldf[name]=finaldf[name].astype(float)
          except:
            continue

        #Auto remove columns if non nummeris >10%, if non numberic < 10% remove row :
        newdf_test=finaldf.copy()
        max_row=len(newdf_test)
        print(max_row)
        for name in newdf_test.columns[1:]:
            a=newdf_test[name]
            non_numeric_column=pd.DataFrame(a)[~pd.DataFrame(a).applymap(np.isreal).all(1)]
            num_row_nonnumeric=len(non_numeric_column)
            #st.write('number of non numeric row in column '+ name + ' before:'+ str(num_row_nonnumeric))
            if num_row_nonnumeric < max_row*0.1:
                list_to_exclude=non_numeric_column.index.values.tolist()
                newdf_test=newdf_test[~newdf_test.index.isin(list_to_exclude)]
                newdf_test[name]=newdf_test[name].astype(float)
            else:
                newdf_test.drop(columns=name,inplace=True)
        #st.write('Check result after process')       
        for name in newdf_test.columns[1:]:
            a=newdf_test[name]
            non_numeric_column=pd.DataFrame(a)[~pd.DataFrame(a).applymap(np.isreal).all(1)]
            num_row_nonnumeric=len(non_numeric_column)
            #st.write('number of non numeric row in column '+ name + ' after:'+ str(num_row_nonnumeric))
        finaldf = newdf_test
        #finaldf.dtypes

        ### set up tolerance dict:USL,LSL,Nominal"""

        #set up tolerance dict:USL,LSL,Nominal
        #Tolerance set up when not drop object type column, not reflect exactly tolerance vs data analysis
        tolerance={}
        for column in df.columns[1:(feature_num+1)]:
            namecolumn_temp=df[column][7] #df: dataframe latest in for loop
            namecolumn_temp=namecolumn_temp.replace('.','')
            tolerance[namecolumn_temp] =[df[column][9],df[column][10],df[column][15]]
        #tolerance

        #Create list of dataframe, combine feature value and USL,LSL, Nominal
        DFdict={}
        for name in finaldf.columns[1:]:
            #print('Chart: ',name)

            #Create series USL:
            USL = tolerance[name][0]
            USL=pd.Series(USL)
            s1=pd.DataFrame()
            for i in range(len(finaldf)):
              s1=pd.concat([s1,USL],axis=0)
            s1.rename(columns={0:'USL'},inplace=True)

            #Create series LSL:
            LSL = tolerance[name][1]
            LSL=pd.Series(LSL)
            s2=pd.DataFrame()
            for i in range(len(finaldf)):
              s2=pd.concat([s2,LSL],axis=0)
            s2.rename(columns={0:'LSL'},inplace=True)

            #Create series nominal:
            nominal = tolerance[name][2]
            nominal=pd.Series(nominal)
            s3=pd.DataFrame()
            for i in range(len(finaldf)):
              s3=pd.concat([s3,nominal],axis=0)
            s3.rename(columns={0:'Nominal'},inplace=True)

            #combine Date,feature, LSL,USL,nominal to finaldataframe (for each feature):
            finaldf['Datef'].reset_index(drop=True,inplace=True)
            finaldf[name].reset_index(drop=True,inplace=True)
            s1.reset_index(drop=True,inplace=True)
            s2.reset_index(drop=True,inplace=True)
            s3.reset_index(drop=True,inplace=True)
            df_temp = pd.concat([finaldf['Datef'],finaldf[name],s1,s2,s3],axis=1)  

            #Create dict of dataframe for each feature columns:
            DFdict[name]=df_temp # Include Date, name, USL, LSL, nominal each parameter(J, I, O or Z...)
            #print(DFdict[name][:3])
        return DFdict

    DFdict=create_dataframe(all_files) # Run function above

    def create_DFdict_final(DFdict):
        DFdict_final={}
        for name in DFdict.keys():
          df=DFdict[name].copy()
          df_group=df.groupby('Datef').mean() # mean value by Date
          df_group.reset_index(inplace=True)

          value_name=df_group.columns[1] #'C-Pitch', 'J', 'I', 'O', 'Z'
          df_group[value_name] = df_group[value_name].round(decimals=3)

          # Create a selection that chooses the nearest point & selects based on x-value
          #UCL,LCL,nominal:
          sigma=3
          df_final = df_group.assign(
                        UCL=df_group[value_name].mean() + df_group[value_name].std()*sigma,
                        LCL=df_group[value_name].mean() - df_group[value_name].std()*sigma,
                        mean=df_group[value_name].mean())
          DFdict_final[name]=df_final
        return (DFdict_final,df_group)

    DFdict_final,df_group=create_DFdict_final(DFdict)
    #------------------Line chart plotly-----------------------------------------#
    st.header("Control chart")

    def line_chart(DFdict_final):
        i=1
        #Layout
        fig = make_subplots(
            rows=len(DFdict_final), cols=1,
            #shared_xaxes=True,
            vertical_spacing=0.05,
            #column_widths=[0.8, 0.2],
            subplot_titles=(list(DFdict_final.keys()))
        )
        for name in DFdict_final.keys(): #also group
            df=DFdict_final[name].copy()
            df=df.sort_values(by=['Datef'])
            for a in df.columns[1:]:
                df[a] = df[a].round(decimals=3)
            df=df.set_index('Datef')
            if start_date != '' and end_date != '':
                df=df[start_date:end_date]  
            #Control chart 1 
            fig.add_trace(go.Scatter(
                                  x=df.index, y=df[name],
                                  mode='lines+markers',
                                  name='mean ' + name,line=dict( color='#4280F5')
                                  ),row=i, col=1
                        )
            #USL, LSL
            fig.add_trace(go.Scatter(x=df.index, y=df['USL'],name='USL '+name, line=dict( color='#FF5733'),mode='lines'),row=i, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['LSL'],name='LSL '+name,line=dict( color='#FF5733'),mode='lines'),row=i, col=1)
            #fig.add_trace(go.Scatter(x=df.index, y=df['Nominal'],name='Nominal '+name,line=dict( color='#FF5733'),mode='lines'),row=i, col=1)
            # UCL, LCL
            fig.add_trace(go.Scatter(x=df.index, y=df['UCL'],name='UCL '+name, line=dict( color='#33C2FF'),mode='lines'),row=i, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['LCL'],name='LCL '+name, line=dict( color='#33C2FF'),mode='lines'),row=i, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['mean'],name='mean '+name, line=dict( color='#33C2FF'),mode='lines'),row=i, col=1)

            #Final layout:
            #fig.update_layout(height=400, width=1400, title_text='Line chart'+name)
            i=i+1

        fig.update_layout(height=230*len(DFdict_final), width=1200, title_text='Line chart')
            #fig.show()
        return fig

    fig_new=line_chart(DFdict_final) 
    return fig_new



#feature_num=5
if __name__ == '__main__':
    app.run_server(debug=True)
    

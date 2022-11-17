# import libraries 

import plotly.graph_objects as go
import dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
import sys
import re
import plotly.express as px
from datetime import date
import datetime
import io
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# define main css and some initial option for dash App

external_stylesheets = [dbc.themes.LUX]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,)
app.title = "ASML Dashboard"
app.config.suppress_callback_exceptions=True


# data gathering and optimisation

pathname = "C:/Users/Administrator/Downloads/ASML-Dashboard/Data/Machine"
machines = ["380", "385", "1385"]


dataframes_list = []

for machine in machines:
    path, dirs, files = next(os.walk(pathname + machine + "/"))
    file_count = len(files)

    for i in range(file_count):
        temp_df = pd.read_csv(pathname + machine + "/" + files[i] , header = None)
        temp_df.columns = ['Date', 'Atomic Timestamp', 'Measurement Action Unique Identifier', 'Measurement Name', 'Operation Type', 'Measurement Type', 'Measurement Context', 'Measurement Action']
        temp_df["Machine"] = machine
        temp_df["Filename"] = files[i]
        
        itemlist = []
        statuslist = []

        for item in temp_df["Measurement Action"]:
            if re.match("^Start.*", item):
                itemlist.append(item[5:])
                statuslist.append(item[:5])
            elif re.match("^Finish.*", item):
                itemlist.append(item[6:])
                statuslist.append(item[:6])
            elif item == "ImageCaptured":
                itemlist.append(item)
                statuslist.append("")
            elif item == "TriggerPrewarningTimerTemp":
                itemlist.append(item)
                statuslist.append("")
            elif item == "FastWaferAlignSelectInitialFields":
                itemlist.append(item)
                statuslist.append("")
            elif item == "SoftwareReleaseInfo":
                itemlist.append(item)
                statuslist.append("")
            else:
                print(item)
        
        
        temp_df["Measurement"] = itemlist
        temp_df["Action"] = statuslist
        
        dataframes_list.append(temp_df)
        
TEMP = pd.concat(dataframes_list)
TEMP.reset_index(drop = True, inplace = True)

# generate start and end event 

TEMP['Finish DateTime'] = pd.to_datetime(TEMP['Date'] + ' ' + TEMP['Atomic Timestamp'])

TEMP['Start DateTime'] = (TEMP[(TEMP['Action'] == "Start") | (TEMP['Action'] == "Finish")].sort_values(by=['Finish DateTime'], ascending=True)
                       .groupby(['Filename' ,'Machine', 'Measurement'])['Finish DateTime'].shift(1))

TEMP['Start Measurement Action Unique Identifier'] = (TEMP[(TEMP['Action'] == "Start") | (TEMP['Action'] == "Finish")].sort_values(by=['Finish DateTime'], ascending=True)
                       .groupby(['Filename' ,'Machine', 'Measurement'])['Measurement Action Unique Identifier'].shift(1))

TEMP['Start Operation Type'] = (TEMP[(TEMP['Action'] == "Start") | (TEMP['Action'] == "Finish")].sort_values(by=['Finish DateTime'], ascending=True)
                       .groupby(['Filename' ,'Machine', 'Measurement'])['Operation Type'].shift(1))

TEMP['Start Measurement Type'] = (TEMP[(TEMP['Action'] == "Start") | (TEMP['Action'] == "Finish")].sort_values(by=['Finish DateTime'], ascending=True)
                       .groupby(['Filename' ,'Machine', 'Measurement'])['Measurement Type'].shift(1))

TEMP['Start Measurement Context'] = (TEMP[(TEMP['Action'] == "Start") | (TEMP['Action'] == "Finish")].sort_values(by=['Finish DateTime'], ascending=True)
                       .groupby(['Filename' ,'Machine', 'Measurement'])['Measurement Context'].shift(1))
                       


# create completed dataframe

ASML = TEMP[['Start DateTime', 'Finish DateTime', 'Start Measurement Action Unique Identifier', 'Measurement Action Unique Identifier', 'Measurement Name', 'Start Operation Type', 'Operation Type', 'Start Measurement Type', 'Measurement Type', 'Start Measurement Context', 'Measurement Context', 'Machine', 'Measurement', 'Action']].copy()
ASML = ASML.rename(columns={"Measurement Action Unique Identifier": "Finish Measurement Action Unique Identifier", "Operation Type": "Finish Operation Type", "Measurement Type": "Finish Measurement Type", "Measurement Context": "Finish Measurement Context", "Action": "Status"})
ASML = ASML[(ASML['Status'] == "Finish") | (ASML['Status'] == "")]
ASML = ASML.drop(ASML[(ASML['Start DateTime'].isna()) & (ASML['Status'] == "Finish")].index)
ASML = ASML.drop(ASML[(ASML['Start DateTime'] == ASML['Finish DateTime']) & (ASML['Status'] == "Finish")].index)
mStatus = (ASML['Status'] == "")
ASML.loc[mStatus, 'Status'] = "OneTimeEvent"
ASML['Start DateTime'] = ASML['Start DateTime'].fillna(ASML['Finish DateTime'])
ASML['Duration'] = (ASML['Finish DateTime']-ASML['Start DateTime']).dt.total_seconds()*1000
conditions = [
    (ASML['Machine'] == "380") | (ASML['Machine'] == "385"),
    (ASML['Machine'] == "1375") | (ASML['Machine'] == "1385")
    ]
groups = ['ADI', 'AEI']
ASML['MachineType'] = np.select(conditions, groups)
ASML[['Start Measurement Action Unique Identifier', 'Start Operation Type']] = ASML[['Start Measurement Action Unique Identifier', 'Start Operation Type']].astype("Int64")
ASML = ASML[(ASML["Measurement"].str.contains(r'.*[\d, _]+$') == False)]

# del temporarty dataframe 

del(temp_df)
del(TEMP)

# print data for menu

sdate = ASML['Start DateTime'].min().date()
fdate = ASML['Finish DateTime'].max().date()
m1 = (ASML['Status'] != "OneTimeEvent" )
measurements = ASML[m1].Measurement.unique()
machinetype = ASML.MachineType.unique()
machinenumber = ASML.Machine.unique()

# tab2 dynamic df #
temp1 = ASML.groupby(by=["MachineType","Measurement"], as_index=False)['Duration'].mean()
temp2 = ASML.groupby(by=["Machine","MachineType","Measurement"], as_index=False)['Duration'].mean()
df9 = temp2.merge(temp1, on=['MachineType','Measurement'], how='left')
del(temp1,temp2)
df9 = df9.rename(columns={"Duration_x": "Duration", "Duration_y": "Average"})
con = [
    (df9['Duration'] < df9['Average']),
    (df9['Duration'] == df9['Average']),
    (df9['Duration'] > df9['Average'])
    ]
pcta = ['Slow', 'Normal', 'Fast']
df9['PerformanceStatus'] = np.select(con, pcta)
df9[' index'] = range(1, len(df9) + 1)
PAGE_SIZE = 10

# Header logo
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/ASML_Holding_N.V._logo.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("Dashboard", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/",
            ),
        ]
    ),
    color="orange",
    dark=True,
    className="mb-4",
)

# Main Layout
app.layout = html.Div([
    navbar,
    dcc.Tabs([
        ######TAB1######
        dcc.Tab(label='Overall View', children=[
      
        dbc.Container([
        html.Br(),
        html.Br(),    
        dbc.Row([
            dbc.Col(html.H1(children='Overall View'), className="mb-2")
        ]),
        dbc.Row([
            dbc.Col(html.H6(children='Visualising trends across machines'), className="mb-4")
        ]),
        dbc.Row([        
            dcc.DatePickerRange(id='t1_date-picker-range', start_date=sdate, end_date=fdate,
            minimum_nights=0, min_date_allowed=sdate,max_date_allowed=fdate,initial_visible_month=sdate)
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(html.H3(children='Machine Comparison',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='Machine Type', className="text-center"), width=6, className="mt-4"),
            dbc.Col(html.H5(children='Machine Specific', className="text-center"), width=6, className="mt-4"),
        ]),
        html.Br(),
        dbc.Row([
        dcc.Dropdown(
        id="t1_machinetype",
        options=[{"label": x, "value": x} for x in machinetype],
        value=machinetype[0],        
        style={'width': '50%', 'align-items': 'left', 'justify-content': 'left', 'whiteSpace': 'pre-wrap'},
        clearable=False
        ),
        dcc.Dropdown(
        id="t1_measurement",
        options=[{"label": x, "value": x} for x in measurements],
        value=measurements[0],        
        style={'width': '50%', 'align-items': 'right', 'justify-content': 'right', 'whiteSpace': 'pre-wrap'},
        clearable=False
        ),
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(dcc.Graph(id='type'), width=6),
            dbc.Col(dcc.Graph(id='specific'), width=6)
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(html.H3(children='Performance',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='Fastest Machines', className="text-center"), width=6, className="mt-4"),
            dbc.Col(html.H5(children='Slowest Machines', className="text-center"), width=6, className="mt-4"),
        ]),
        html.Br(),
        dbc.Row([
        dcc.Dropdown(
        id="t1_fsmeasurement",
        options=[{"label": x, "value": x} for x in measurements],
        value=measurements[0],        
        style={'width': '100%', 'align-items': 'center', 'justify-content': 'center', 'whiteSpace': 'pre-wrap'},
        clearable=False
        ),
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(dcc.Graph(id='fastest'), width=6),
            dbc.Col(dcc.Graph(id='slowest'), width=6)
        ]),
        html.Br(),
         dbc.Row([
            dbc.Col(dbc.Card(html.H3(children='Deterioration',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='Deterioration Gauge', className="text-center"), width=6, className="mt-4"),
            dbc.Col(html.H5(children='Deterioration Overtime', className="text-center"), width=6, className="mt-4"),
        ]),
        html.Br(),
        dbc.Row([
        dcc.Dropdown(
        id="t1_dmeasurement",
        options=[{"label": x, "value": x} for x in measurements],
        value=measurements[0],        
        style={'width': '100%', 'align-items': 'right', 'justify-content': 'right', 'whiteSpace': 'pre-wrap'},
        clearable=False
        ),
        dcc.Dropdown(
        id="t1_machinenumbers",
        options=[{"label": x, "value": x} for x in machinenumber],
        value=machinenumber[0],        
        style={'width': '50%', 'align-items': 'left', 'justify-content': 'left', 'whiteSpace': 'pre-wrap'},
        clearable=False
        ),
        ]),
        html.Br(),
        dbc.Row([
        dbc.Col(dcc.Graph(id='gauge'), width=4),
        dbc.Col(dcc.Graph(id='dot'), width=8)
        ]),
        html.Br(),
    
        ])
        ]),
        ######TAB2######
        dcc.Tab(label='Machine Type Overview', children=[
        dbc.Container([
        html.Br(),    
        html.Br(), 
        dbc.Row([
            dbc.Col(html.H1(children='Machine Type Overview'), className="mb-2")
        ]),
        dbc.Row([
            dbc.Col(html.H6(children='Visualising trends across machines'), className="mb-4")
        ]),
        dbc.Row([        
            dcc.DatePickerRange(id='t2_date-picker-range', start_date=sdate, end_date=fdate,
            minimum_nights=0, min_date_allowed=sdate,max_date_allowed=fdate,initial_visible_month=sdate)
        ]),        
        dbc.Row([
            dbc.Col(dbc.Card(html.H3(children='Overview Performance Table',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
        ]), 
        dcc.Dropdown(
        id="t2_machinenumbers",
        options=[{"label": x, "value": x} for x in machinenumber],
        value=machinenumber[0],        
        style={'width': '50%', 'align-items': 'left', 'justify-content': 'left', 'whiteSpace': 'pre-wrap'},
        clearable=False
        ), 
        html.Br(),        
        dash_table.DataTable(
        id='table-sorting-filtering',
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in df9.loc[:,['MachineType','Machine','Measurement','Duration','PerformanceStatus']]
        ],
        page_current= 0,
        page_size= PAGE_SIZE,
        page_action='custom',

        filter_action='custom',
        filter_query='',

        sort_action='custom',
        sort_mode='multi',
        sort_by=[]
        ),
        dbc.Row([
            dbc.Col(dbc.Card(html.H3(children='Comparison of TPT Point',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
        ]),        
        dbc.Row([
        dcc.Dropdown(
        id="t2_measurement",
        options=[{"label": x, "value": x} for x in measurements],
        value=measurements[0],        
        style={'width': '100%', 'align-items': 'center', 'justify-content': 'center', 'whiteSpace': 'pre-wrap'},
        clearable=False
        ),
        html.Br(),
        html.Br(),  
        html.Br(), 
        ]), 
        dbc.Row([
        dbc.Col(dcc.Graph(id='tptpointcompare'), width=12)
        ]),
        html.Br(),    
        html.Br()        
        ])        
        ]),
        ######TAB3######
        dcc.Tab(label='TPT Point Overview', children=[           
        dbc.Container([
        html.Br(),    
        html.Br(),         
        dbc.Row([
            dbc.Col(html.H1(children='TPT Point Overview'), className="mb-2")
        ]),
        dbc.Row([
            dbc.Col(html.H6(children='Visualising trends across machines'), className="mb-4")
        ]),
        dbc.Row([        
            dcc.DatePickerRange(id='t3_date-picker-range', start_date=sdate, end_date=fdate,
            minimum_nights=0, min_date_allowed=sdate,max_date_allowed=fdate,initial_visible_month=sdate)
        ]),
        dbc.Row([
            dbc.Col(dbc.Card(html.H3(children='TPT Point Comparison',
                                 className="text-center text-light bg-dark"), body=True, color="dark")
        , className="mt-4 mb-4")
        ]),
        dcc.Slider(
        id='t3_nword',
        min=10,
        max=200,
        step=10,
        value=100,
        tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Img(id='wordcloud'),
        dbc.Row([
            dbc.Col(html.H5(children='Fastest Machines', className="text-center"), width=6, className="mt-4"),
            dbc.Col(html.H5(children='Slowest Machines', className="text-center"), width=6, className="mt-4"),
        ]),
        html.Br(),
        dbc.Row([
        dcc.Dropdown(
        id="t3_measurement",
        options=[{"label": x, "value": x} for x in measurements],
        value=measurements[0],        
        style={'width': '100%', 'align-items': 'center', 'justify-content': 'center', 'whiteSpace': 'pre-wrap'},
        clearable=False
        ),
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(dcc.Graph(id='t3_fastest'), width=6),
            dbc.Col(dcc.Graph(id='t3_slowest'), width=6)
        ]),        
        html.Br(),    
        html.Br()
        ])
        ]),  
        ######TAB4#######
        dcc.Tab(label='Machine Specific Overview', children=[
            dbc.Container([
            html.Br(),
            html.Br(),            
            dbc.Row([
                dbc.Col(html.H1(children='Machine Specific Overview'), className="mb-2")
            ]),
            dbc.Row([
                dbc.Col(html.H6(children='Visualising trends across machines'), className="mb-4")
            ]),
            dbc.Row([        
                dcc.DatePickerRange(id='t4_date-picker-range', start_date=sdate, end_date=fdate,
                minimum_nights=0, min_date_allowed=sdate,max_date_allowed=fdate,initial_visible_month=sdate)
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(html.H3(children='TPT Point Comparison',
                                     className="text-center text-light bg-dark"), body=True, color="dark")
            , className="mt-4 mb-4")
            ]),
            dbc.Row([
            dcc.Dropdown(
            id="t4_measurement",
            options=[{"label": x, "value": x} for x in measurements],
            value=measurements[0],        
            style={'width': '50%', 'align-items': 'left', 'justify-content': 'left', 'whiteSpace': 'pre-wrap'},
            clearable=False
            ),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Graph(id='m380'), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='m385'), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='m1385'), width=12)
            ]),
            html.Br(),    
            html.Br()    
            ])
        ]),
    ]),
    dbc.Row(
            [
            html.P('Developed by Group 1 || Nicolas - Linh - Kingsley || 12.2021', style={'font-style': 'italic'})
            ], justify="center", align="center"
            )
])

operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3

@app.callback([Output("type", "figure"), 
    Output('specific', 'figure'),
    Output("fastest", "figure"), 
    Output('slowest', 'figure'),
    Output('gauge', 'figure'),
    Output('dot', 'figure'),
    Output('table-sorting-filtering', 'data'),
    Output('tptpointcompare', 'figure'),
    Output('wordcloud', 'src'),
    Output("t3_fastest", "figure"), 
    Output('t3_slowest', 'figure'),
    Output("m380", "figure"), 
    Output('m385', 'figure'),
    Output("m1385", "figure")],
    [Input("t1_date-picker-range", "start_date"), Input("t1_date-picker-range", "end_date"),
     Input("t1_machinetype", "value"), Input("t1_measurement", "value"), 
     Input("t1_fsmeasurement", "value"), 
     Input("t1_machinenumbers", "value"), Input("t1_dmeasurement", "value"),
     Input("t2_date-picker-range", "start_date"), Input("t2_date-picker-range", "end_date"),
     Input('table-sorting-filtering', "page_current"), Input('table-sorting-filtering', "page_size"), Input('table-sorting-filtering', 'sort_by'), Input('table-sorting-filtering', 'filter_query'), Input("t2_machinenumbers", "value"), Input("t2_measurement", "value"),
     Input("t3_date-picker-range", "start_date"), Input("t3_date-picker-range", "end_date"),
     Input('t3_nword', 'value'), Input("t3_measurement", "value"), 
     Input("t4_date-picker-range", "start_date"), Input("t4_date-picker-range", "end_date"),
     Input("t4_measurement", "value")])
        
        
def update_chart(t1_start_date,t1_end_date,t1_MachineType,t1_Measurement,t1_FSMeasurement,t1_MachineNumber,t1_DMeasurement,t2_start_date,t2_end_date,page_current,page_size,sort_by,filter,t2_machinenumbers,t2_measurement,t3_start_date,t3_end_date,t3_Word,t3_measurement,t4_start_date,t4_end_date,t4_Measurement):

#################################################TAB1########################################################################
    mask = (ASML['Start DateTime'] > t1_start_date) & (ASML['Finish DateTime'] < pd.to_datetime(t1_end_date) + datetime.timedelta(1)) & (ASML["MachineType"] == t1_MachineType)
    mask2 = (ASML['Start DateTime'] > t1_start_date) & (ASML['Finish DateTime'] < pd.to_datetime(t1_end_date) + datetime.timedelta(1)) & (ASML["Measurement"] == t1_Measurement)
    fig = px.bar(ASML[mask].groupby(by=["Machine"], as_index=False).agg({"Duration": "mean"}), 
                 x="Machine", y="Duration",
                 labels={"Duration": "Duration in milliseconds","Machine": "Machine Number"},
                 title="Average Duration Over All Measurements",
                 template="plotly_dark"
                 )
    fig2 = px.bar(ASML[mask2].groupby(by=["Machine"], as_index=False).agg({"Duration": "mean"}), 
                 x="Machine", y="Duration",
                 labels={"Duration": "Duration in milliseconds","Machine": "Machine Number"},
                 title="Average Duration Per Measurement",
                 template="plotly_dark"
                 )
    mask3 = (ASML['Start DateTime'] > t1_start_date) & (ASML['Finish DateTime'] < pd.to_datetime(t1_end_date) + datetime.timedelta(1)) & (ASML["Measurement"] == t1_FSMeasurement)            
    df1 = ASML[mask3].groupby(by=["Machine","Measurement"], as_index=False)['Duration'].mean().nsmallest(2, 'Duration')
    df2 = ASML[mask3].groupby(by=["Machine","Measurement"], as_index=False)['Duration'].mean().nlargest(2, 'Duration')
    fig3 =  px.bar(df1, x="Duration", y="Machine",
            labels={"Duration": "Average Duration in milliseconds","Machine": "Machine Number"},
            title="Top 2 Fastest Machines Per Measurement",
            template="plotly_dark",
            orientation='h')
    fig4 =  px.bar(df2, x="Duration", y="Machine", 
            labels={"Duration": "Average Duration in milliseconds","Machine": "Machine Number"},
            title="Top 2 Slowest Machines Per Measurement",
            template="plotly_dark",
            orientation='h')
    mask4 = (ASML['Start DateTime'] > t1_start_date) & (ASML['Finish DateTime'] < pd.to_datetime(t1_end_date) + datetime.timedelta(1)) & (ASML["Machine"] == t1_MachineNumber) & (ASML["Measurement"] == t1_DMeasurement)    
    df3 = ASML[mask4].groupby(by=[pd.to_datetime(ASML['Finish DateTime']).dt.floor('d'),"Machine","Measurement"])['Duration'].mean().reset_index()
    df3 = df3.rename(columns={"Finish DateTime": "Date"})
    df3['Deterioration'] = abs(100 - 100 / df3.sort_values(by=['Date','Machine'], ascending=False).groupby(['Machine'])['Duration'].shift(-1) * df3['Duration']) 
    try:
        value1 = df3.sort_values(by=['Date'], ascending=False)["Deterioration"].values[0]
    except:
        value1 = 0
    fig5 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = value1,
    title = {'text': "Percentage"},
    gauge = {'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
             'bar': {'color': "#636EFA"},
             'steps' : [
                 {'range': [0, 50], 'color': "lightcoral"},
                 {'range': [50, 100], 'color': "indianred"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 20}}))
    fig5.update_layout(title="Average Deterioration Percentage<br>(Measured Per Day)", paper_bgcolor = "black", font = {'color': "white"}, height=500)
    
    mask5 = (ASML['Start DateTime'] > t1_start_date) & (ASML['Finish DateTime'] < pd.to_datetime(t1_end_date) + datetime.timedelta(1)) & (ASML["Measurement"] == t1_DMeasurement)    
    df4 = ASML[mask5].groupby(by=[pd.to_datetime(ASML['Finish DateTime']).dt.floor('min'),"Machine","Measurement"])['Duration'].mean().reset_index()
    df4['Deterioration'] = abs(100 - 100 / df4.sort_values(by=['Finish DateTime','Machine'], ascending=False).groupby(['Machine'])['Duration'].shift(-1) * df4['Duration']) 
    df4['Finish DateTime'] = df4['Finish DateTime'].astype(str).str.replace('-','/')
    fig6 = px.line(df4[df4.Deterioration.notnull()], x='Finish DateTime', y='Deterioration', 
                   color='Machine', markers=True, 
                   facet_col="Machine",
                   height=500,
                   category_orders={"Machine": ["380", "385","1385"]},
                   title="Average Deterioration Percentage (Measured Per Minute)",
                   template="plotly_dark",)
    fig6.update_xaxes(rangeslider_visible=True,matches=None, showticklabels=True, tickangle=40, tickfont=dict(size=10))    
    for axis in fig6.layout:
        if type(fig6.layout[axis]) == go.layout.YAxis:
            fig6.layout[axis].title.text = ''
        if type(fig6.layout[axis]) == go.layout.XAxis:
            fig6.layout[axis].title.text = ''
    fig6.update_layout(
        annotations = list(fig6.layout.annotations) + 
        [go.layout.Annotation(
                x=-0.09,
                y=0.5,
                font=dict(
                    size=14, color = 'White'
                ),
                showarrow=False,
                text="Deterioration %",
                textangle=-90,
                xref="paper",
                yref="paper"
            )
        ] +
        [go.layout.Annotation(
                x=0.5,
                y=-1,
                font=dict(
                    size=14, color = 'White'
                ),
                showarrow=False,
                text="Date & Time Slicer",
                textangle=-0,
                xref="paper",
                yref="paper"
            )
        ]
    )         
#################################################TAB3########################################################################    
    mask7 = (ASML['Start DateTime'] > t3_start_date) & (ASML['Finish DateTime'] < pd.to_datetime(t3_end_date) + datetime.timedelta(1))
    df6 = ASML[mask].groupby(by=["Measurement"], as_index=False).agg({"Duration": "mean"})
    data = df6.set_index('Measurement').to_dict()['Duration']
    #df6 = ASML[mask].groupby(by=["Measurement"], as_index=True).agg({"Measurement": "count"}).rename(columns={"Measurement": "Count"}).reset_index()
    #data = df6.set_index('Measurement').to_dict()['Count']
    wc = WordCloud(width=1050, height=500, max_words=t3_Word).generate_from_frequencies(data)
    plt.figure(figsize=(10.5, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    buf = io.BytesIO() # in-memory files
    plt.savefig(buf, format = "png") # save to the above file object
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements    
    mask8 = (ASML['Start DateTime'] > t3_start_date) & (ASML['Finish DateTime'] < pd.to_datetime(t3_end_date) + datetime.timedelta(1)) & (ASML["Measurement"] == t3_measurement)            
    df7 = ASML[mask8].groupby(by=["Machine","Measurement"], as_index=False)['Duration'].mean().nsmallest(2, 'Duration')
    df8 = ASML[mask8].groupby(by=["Machine","Measurement"], as_index=False)['Duration'].mean().nlargest(2, 'Duration')
    fig10 =  px.bar(df7, x="Duration", y="Machine",
            labels={"Duration": "Average Duration in milliseconds","Machine": "Machine Number"},
            title="Top 2 Fastest Machines Per Measurement",
            template="plotly_dark",
            orientation='h')
    fig11 =  px.bar(df8, x="Duration", y="Machine", 
            labels={"Duration": "Average Duration in milliseconds","Machine": "Machine Number"},
            title="Top 2 Slowest Machines Per Measurement",
            template="plotly_dark",
            orientation='h')   
#################################################TAB4########################################################################
    mask6 = (ASML['Start DateTime'] > t4_start_date) & (ASML['Finish DateTime'] < pd.to_datetime(t4_end_date) + datetime.timedelta(1)) & (ASML["Measurement"] == t4_Measurement)    
    df5 = ASML[mask6][['Finish DateTime','Machine','Duration']]
    df5['Finish DateTime'] = df5['Finish DateTime'].astype(str).str.replace('-','/')
    fig7 = px.line(df5[df5['Machine'] == '380'], x='Finish DateTime', y='Duration', markers=True,
                    height=500,
                    title="Machine 380",
                    template="plotly_dark",)
    fig8 = px.line(df5[df5['Machine'] == '385'], x='Finish DateTime', y='Duration', markers=True, 
                    color_discrete_sequence=['#EF553B'], height=500,
                    title="Machine 385",
                    template="plotly_dark",)
    fig9 = px.line(df5[df5['Machine'] == '1385'], x='Finish DateTime', y='Duration', markers=True,
                    color_discrete_sequence=['#03A77B'],height=500,
                    title="Machine 1385",
                    template="plotly_dark",)
#################################################TAB2########################################################################    
    filtering_expressions = filter.split(' && ')
    mask7 = (df9["Machine"] == t2_machinenumbers)   
    dff = df9[mask7] 
    mask8 = (ASML['Start DateTime'] > t2_start_date) & (ASML['Finish DateTime'] < pd.to_datetime(t2_end_date) + datetime.timedelta(1)) & (ASML["Machine"] == t2_machinenumbers) & (ASML["Measurement"] == t2_measurement)        
    df10 = ASML[mask8][['Finish DateTime','Machine','Duration']]
    df10['Finish DateTime'] = df10['Finish DateTime'].astype(str).str.replace('-','/')    
    fig12 = px.line(df10, x='Finish DateTime', y='Duration', markers=True,
                    height=500,
                    title="TPT Point Acquire on Machine" + t2_machinenumbers,
                    template="plotly_dark",)    
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff = dff.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )

    page = page_current
    size = page_size  
    
    return fig, fig2, fig3, fig4, fig5, fig6, dff.iloc[page * size: (page + 1) * size].to_dict('records'), fig12, "data:image/png;base64,{}".format(data), fig10, fig11, fig7, fig8, fig9 

if __name__ == '__main__':
    app.run_server(debug=True)
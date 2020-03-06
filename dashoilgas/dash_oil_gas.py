#!/usr/bin/env python
# coding: utf-8

# ## Like Oil and Gas
# #### Yet another version of the Dash *New York Oil and Gas* Gallery app, that simplifies the callback boilerplate, and determines the visual layout using only css.
# 
# ![alt text](dash_oil_gas.png "")
# 
# #### To Run:
# Run all the cells in this notebook (```dash_oil_gas.ipynb```).
# 
# #### How it works:
# 1. Simple css grid layouts
#     * The workbook creates a Dash.app using Dash.html and Dash.dcc components.  The css file assets/oilgas.css defines the layout of the page by assigning ```display:grid``` css definitions at both the html id and html class level. 
#     * Below, you can see the basic layout, which consists of 3 grid rows.  Each row has several columns, which might themselves have nested rows and columns.
# 
# 
# 2. The ```DashLink``` class defines callbacks that link the components:
#     * The ```DashLink``` class provides methods for defining any kind of callback between any set of Dash html or dcc components.  You will not find separate ```@app.callback``` definitions in the code.  There is a single reusable callback closure that allows you to create call backs without copying and pasting Dash's callback boilerplate.  
#     * See the ```DashLink``` constructor and the ```DashLink.callback()``` method.
# 
# 
# ## Grid Layout of Page
# ___
# ### row 1: Title
# * dash_logo
# * title
# * dash_learn_more
# ___
# ### row 2: Filters and Year Graph
# * column 1: **Filters**
#     * filter_construct_date
#     * filter_well_status
#     * filter_well_type
# * column 2: **Panels and Graph**
#     * row 1: *Panels*
#         * no_wells
#         * mcf_gas
#         * bbl_oil
#         * bbl_water
#     * row 2: *Graphs*
#         * wells_per_year_graph
# ___
# ### row 3: Well satellite map and well type pie chart
# * column 1: **Map and Graph**
#     * satellite_map
#     * prod_summary_graph
# ___
# ### bottom_div
# ___
# 
# ## To run as a stand-alone web app:
# 
# #### A version of this notebook has been converted to a .py module using:
# ```!jupyter nbconvert --to script dash_oil_gas.ipynb```

# In[1]:


import sys,os
if  not os.path.abspath('./') in sys.path:
    sys.path.append(os.path.abspath('./'))
if  not os.path.abspath('../') in sys.path:
    sys.path.append(os.path.abspath('../'))

import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from plotly.graph_objs.layout import Margin#,Font
from dash.dependencies import Input, Output,State
from dash.exceptions import PreventUpdate
import dash_table
import pandas as pd
import numpy as np
import json
import logging
import datetime
import functools
import random
import inspect
from pandasql import sqldf
import datetime,base64,io,pytz
import datetime as dt
import collections as cole
import pickle
import pathlib
import copy


# ### First define a url_base_pathname, used later in the app definition.
# The url_base_pathname is useful if you are accessing the app via something like an nginx proxy.

# In[2]:


# url_base_pathname = None
url_base_pathname = '/oilgas/'


# In[3]:


DEFAULT_LOG_PATH = './logfile.log'
DEFAULT_LOG_LEVEL = 'INFO'

def init_root_logger(logfile=DEFAULT_LOG_PATH,logging_level=DEFAULT_LOG_LEVEL):
    level = logging_level
    if level is None:
        level = logging.DEBUG
    # get root level logger
    logger = logging.getLogger()
    if len(logger.handlers)>0:
        return logger
    logger.setLevel(logging.getLevelName(level))

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)   
    return logger


# In[4]:


logger = init_root_logger(logging_level='DEBUG')

def stop_callback(errmess,logger=None):
    m = "****************************** " + errmess + " ***************************************"     
    if logger is not None:
        logger.debug(m)
    raise PreventUpdate()


# In[5]:


def plotly_plot(df_in,x_column,plot_title=None,
                y_left_label=None,y_right_label=None,
                bar_plot=False,figsize=(16,10),
                number_of_ticks_display=20,
                yaxis2_cols=None,
                x_value_labels=None):
    ya2c = [] if yaxis2_cols is None else yaxis2_cols
    ycols = [c for c in df_in.columns.values if c != x_column]
    # create tdvals, which will have x axis labels
    td = list(df_in[x_column]) 
    nt = len(df_in)-1 if number_of_ticks_display > len(df_in) else number_of_ticks_display
    spacing = len(td)//nt
    tdvals = td[::spacing]
    tdtext = tdvals
    if x_value_labels is not None:
        tdtext = [x_value_labels[i] for i in tdvals]
    
    # create data for graph
    data = []
    # iterate through all ycols to append to data that gets passed to go.Figure
    for ycol in ycols:
        if bar_plot:
            b = go.Bar(x=td,y=df_in[ycol],name=ycol,yaxis='y' if ycol not in ya2c else 'y2')
        else:
            b = go.Scatter(x=td,y=df_in[ycol],name=ycol,yaxis='y' if ycol not in ya2c else 'y2')
        data.append(b)

    # create a layout
    layout = go.Layout(
        title=plot_title,
        xaxis=dict(
            ticktext=tdtext,
            tickvals=tdvals,
            tickangle=45,
            type='category'),
        yaxis=dict(
            title='y main' if y_left_label is None else y_left_label
        ),
        yaxis2=dict(
            title='y alt' if y_right_label is None else y_right_label,
            overlaying='y',
            side='right'),
        margin=Margin(
            b=100
        )        
    )

    fig = go.Figure(data=data,layout=layout)
    fig.update_layout(
        title={
            'text': plot_title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    return fig


# In[6]:


DEFAULT_TIMEZONE = 'US/Eastern'


# In[7]:


# ************************* define useful factory methods *****************

def parse_contents(contents):
    '''
    app.layout contains a dash_core_component object (dcc.Store(id='df_memory')), 
      that holds the last DataFrame that has been displayed. 
      This method turns the contents of that dash_core_component.Store object into
      a DataFrame.
      
    :param contents: the contents of dash_core_component.Store with id = 'df_memory'
    :returns pandas DataFrame of those contents
    '''
    c = contents.split(",")[1]
    c_decoded = base64.b64decode(c)
    c_sio = io.StringIO(c_decoded.decode('utf-8'))
    df = pd.read_csv(c_sio)
    # create a date column if there is not one, and there is a timestamp column instead
    cols = df.columns.values
    cols_lower = [c.lower() for c in cols] 
    if 'date' not in cols_lower and 'timestamp' in cols_lower:
        date_col_index = cols_lower.index('timestamp')
        # make date column
        def _extract_dt(t):
            y = int(t[0:4])
            mon = int(t[5:7])
            day = int(t[8:10])
            hour = int(t[11:13])
            minute = int(t[14:16])
            return datetime.datetime(y,mon,day,hour,minute,tzinfo=pytz.timezone(DEFAULT_TIMEZONE))
        # create date
        df['date'] = df.iloc[:,date_col_index].apply(_extract_dt)
    return df

def make_df(dict_df):
    if type(dict_df)==list:
        if type(dict_df[0])==list:
            dict_df = dict_df[0]
        return pd.DataFrame(dict_df,columns=dict_df[0].keys())
    else:
        return pd.DataFrame(dict_df,columns=dict_df.keys())

class BadColumnsException(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)



def create_dt_div(dtable_id,df_in=None,
                  columns_to_display=None,
                  editable_columns_in=None,
                  title='Dash Table',logger=None,
                  title_style=None):
    '''
    Create an instance of dash_table.DataTable, wrapped in an dash_html_components.Div
    
    :param dtable_id: The id for your DataTable
    :param df_in:     The pandas DataFrame that is the source of your DataTable (Default = None)
                        If None, then the DashTable will be created without any data, and await for its
                        data from a dash_html_components or dash_core_components instance.
    :param columns_to_display:    A list of column names which are in df_in.  (Default = None)
                                    If None, then the DashTable will display all columns in the DataFrame that
                                    it receives via df_in or via a callback.  However, the column
                                    order that is displayed can only be guaranteed using this parameter.
    :param editable_columns_in:    A list of column names that contain "modifiable" cells. ( Default = None)
    :param title:    The title of the DataFrame.  (Default = Dash Table)
    :param logger:
    :param title_style: The css style of the title. Default is dgrid_components.h4_like.
    '''
    # create logger 
    lg = init_root_logger() if logger is None else logger
    
    lg.debug(f'{dtable_id} entering create_dt_div')
    
    # create list that 
    editable_columns = [] if editable_columns_in is None else editable_columns_in
    datatable_id = dtable_id
    dt = dash_table.DataTable(
        page_current= 0,
        page_size= 100,
        filter_action='none', # 'fe',
#         fixed_rows={'headers': True, 'data': 0},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left',
            } for c in ['symbol', 'underlying']
        ],

        style_as_list_view=False,
        style_table={
#             'maxHeight':'450px','overflowX': 'scroll','overflowY':'scroll'
            'overflowY':'scroll'
        } ,
        
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        
        editable=True,
        css=[{"selector": "table", "rule": "width: 100%;"}],
        id=datatable_id
    )
    if df_in is None:
        df = pd.DataFrame({'no_data':[]})
    else:
        df = df_in.copy()
        if columns_to_display is not None:
            if any([c not in df.columns.values for c in columns_to_display]):
                m = f'{columns_to_display} are missing from input data. Your input Csv'
                raise BadColumnsException(m)           
            df = df[columns_to_display]
            
    dt.data=df.to_dict('rows')
    dt.columns=[{"name": i, "id": i,'editable': True if i in editable_columns else False} for i in df.columns.values]                    
    lg.debug(f'{dtable_id} exiting create_dt_div')
    return dt


# In[8]:


class_converters = {
    dcc.Checklist:lambda v:v,
    dcc.DatePickerRange:lambda v:v,
    dcc.DatePickerSingle:lambda v:v,
    dcc.Dropdown:lambda v:v,
    dcc.Input:lambda v:v,
    dcc.Markdown:lambda v:v,
    dcc.RadioItems:lambda v:v,
    dcc.RangeSlider:lambda v:v,
    dcc.Slider:lambda v:v,
    dcc.Store:lambda v:v,
    dcc.Textarea:lambda v:v,
    dcc.Upload:lambda v:v,
}

html_members = [t[1] for t in inspect.getmembers(html)]
dcc_members = [t[1] for t in inspect.getmembers(dcc)]
all_members = html_members + dcc_members

class DashLink():
    def __init__(self,in_tuple_list, out_tuple_list,io_callback=None,
                 state_tuple_list= None,logger=None):
        self.logger = init_root_logger() if logger is None else logger
        _in_tl = [(k.id if type(k) in all_members else k,v) for k,v in in_tuple_list]
        _out_tl = [(k.id if type(k) in all_members else k,v) for k,v in out_tuple_list]
        self.output_table_names = _out_tl
        
        self.inputs = [Input(k,v) for k,v in _in_tl]
        self.outputs = [Output(k,v) for k,v in _out_tl]
        
        self.states = [] 
        if state_tuple_list is not None:
            _state_tl = [(k.id if type(k) in all_members else k,v) for k,v in state_tuple_list]
            self.states = [State(k,v) for k,v in _state_tl]
        
        self.io_callback = lambda input_list:input_list[0] 
        if io_callback is not None:
            self.io_callback = io_callback
                       
    def callback(self,theapp):
        @theapp.callback(
            self.outputs,
            self.inputs,
            self.states
            )
        def execute_callback(*inputs_and_states):
            l = list(inputs_and_states)
            if l is None or len(l)<1 or l[0] is None:
                stop_callback(f'execute_callback no data for {self.output_table_names}',self.logger)
            ret = self.io_callback(l)
            return ret if type(ret) is list else [ret]
        return execute_callback
        


# In[9]:


class DashApp():
    def __init__(self):
        self.all_component_tuples = []
        self.all_dash_links = []
        
    def act_append(self,component_list,css_layout=None):
        cssl = css_layout
        if cssl is None:
            cssl = ' '.join(['1fr' for _ in component_list])
        component_already_exists = False
        for comp in component_list:
            for act in self.all_component_tuples:
                for existing_component in act[0]:
                    if comp.id==existing_component.id:
                        component_already_exists = True
                        break
        if not component_already_exists:
            new_tuple = (component_list,cssl)
            self.all_component_tuples.append(new_tuple)
        else:
            print(f'act_append component {comp.id} already in all_component_tuples')

    def make_component_and_css_lists(self):
        comp_list = []
        css_list = []
        for act in self.all_component_tuples:
            comp_list.extend(act[0])
            css_list.append(act[1])
        return comp_list,css_list

    def adl_append(self,dashlink):
        link_already_in_list = False
        for otn in dashlink.output_table_names:
            for adl in self.all_dash_links:
                for adl_otn in adl.output_table_names:
                    if otn == adl_otn:
                        link_already_in_list = True
                        break
        if not link_already_in_list:
            self.all_dash_links.append(dashlink)
        else:
            print(f'adl_append output {otn} already in output in all_dask_links')

            


# ### Create the global DataFrames that hold information about wells

# #### Read in the 2 main csv files that hold:
# 1. a DataFrame (```df_well_info```) with one row per API_WellNo, that describes the well, and
# 2. a DataFrame (```df_well_production_by_year```) with one row per API_WellNo and per year, describing production amounts 

# In[10]:


df_well_info = pd.read_csv('oil_gas_data/df_well_info.csv')
df_well_production_by_year = pd.read_csv('oil_gas_data/df_well_production_by_year.csv')


# ### Create DataFrames and options dictionaries for:
# 1. well status (```df_well_status``` and ```well_status_options```)
# 2. well type (```df_well_status``` and ```well_status_options```)
# 3. well color (color used in pie charts ```df_well_status``` and ```well_status_options```)
# 

# In[11]:


df_well_status = df_well_info[['Well_Status','wstatus']].drop_duplicates().sort_values('Well_Status')
df_well_status.index = list(range(len(df_well_status)))
well_status_options = [{"label": wt[1], "value": wt[0]} for wt in df_well_status.values]

df_well_type = df_well_info[['Well_Type','wtype']].drop_duplicates().sort_values('Well_Type')
df_well_type.index = list(range(len(df_well_type)))
well_type_options = [{"label": wt[1], "value": wt[0]} for wt in df_well_type.values]

df_well_color = df_well_info[['Well_Type','wcolor']].drop_duplicates().sort_values('Well_Type')
df_well_color.index = list(range(len(df_well_color)))
well_color_options = [{"label": wt[1], "value": wt[0]} for wt in df_well_color.values]


# In[12]:


dap = DashApp()


# ### ```drg``` creates a div which wraps components in css panels 
# (*see the oil_gas.css*)
# * ```rpanel```:  css that should emulated a raised panel
# * ```rpanelnc```: css that should emulate a raised panel with not background color 
# 

# In[13]:


pn = 'rpanel' # see the oil_gas.css file for how this css class is defined
pnnc = 'rpanelnc'

def dgr(div_id,children,parent_class=None,child_class=None):
    return html.Div([html.Div(c,className=child_class) for c in children],id=div_id,className=parent_class)


# ### Row 1: Title

# In[14]:


# Define the static components
# ********************** plotly logo ****************************
imgfolder = '' if url_base_pathname is None else url_base_pathname
img_logo = html.Img(id='img_logo',src=imgfolder + "/assets/dash-logo.png",className='plogo')
# ********************** title div ********************************
title = html.Div(
    [html.H3("New York Oil And Gas",className='ogtitle'),
     html.H5("Production Overview",className='ogtitle')],id='title')
# *****You ************* link to plotly info ***********************
adiv = html.A([html.Button(["Learn More"],className='ogabutton')],id='adiv',href="https://plot.ly/dash/pricing",className='adiv')
r1 = dgr('r1',[img_logo,title,adiv],child_class=pnnc)


# ### Row 2 - Column 1:  Filtering using a slider, radioitems and dropdowns

# In[15]:


# ********************* slider *********************************
slider = dcc.RangeSlider(
        id='yr_slider',
        className='r2_margin',
        min=df_well_info.Year_Well_Completed.min(),
        max=df_well_info.Year_Well_Completed.max(),
        value=[df_well_info.Year_Well_Completed.min(), df_well_info.Year_Well_Completed.max()]    
)
slider_div = html.Div([
    'Filter by construction date (or select range in histogram):',
    html.P(),slider],id='slider_div')

# ********************* well status radio *****************************
rs_op=[{'label': 'All', 'value': 'all'},{'label': 'Active only', 'value': 'active'},{'label': 'Customize', 'value': 'custom'}]
radio_status = dcc.RadioItems(
    id='radio_status',
    options = rs_op,
    value='all'
)
radio1 = html.Div([radio_status,'Filter by well status:',html.P()],
                  id='radio1',className='r2_margin')

# ********************* well status dropdown *****************************
ws_keys = [wso['value'] for wso in well_status_options]
dropdown_status = dcc.Dropdown(
    id='dropdown_status',
    options=well_status_options,value=ws_keys,multi=True)
dropdown1 = html.Div(dropdown_status,className='d1_margin',id='dropdown1')

def build_link_radio1_dropdown1(input_list):
    radio_status =  input_list[0]
    if radio_status.lower()=='all':
        ret = well_status_options
    elif radio_status.lower()=='active':
        i = ws_keys.index('AC')
        ret = [well_status_options[i]]
    else:
        ret = {}
    return [ret]
link_radio1_dropdown1 = DashLink(
    [(radio_status,'value')],
    [(dropdown_status,'options')],build_link_radio1_dropdown1)
dap.adl_append(link_radio1_dropdown1)
link_radio1_dropdown1_value = DashLink(
    [(radio_status,'value')],
    [(dropdown_status,'value')],
    lambda input_list:[[wso['value'] for wso in build_link_radio1_dropdown1(input_list)[0]]])
dap.adl_append(link_radio1_dropdown1_value)

# ********************* well type radio *********************************
rt_op=[{'label': 'All', 'value': 'all'},{'label': 'Productive only', 'value': 'productive'},{'label': 'Customize', 'value': 'custom'}]
radio_type = dcc.RadioItems(id='radio_type',options=rt_op,
                               value='all'
)
radio2 = html.Div([radio_type],id='radio2',className='r2_margin')

# ********************* dropdown for well type *********************************
wt_keys = [wto['value'] for wto in well_type_options]
dropdown_type = dcc.Dropdown(id='dropdown_type',options=well_type_options,value=wt_keys,multi=True)
dropdown2 = html.Div(dropdown_type,className='d1_margin',id='dropdown2')

def build_link_radio2_dropdown2(input_list):
    radio_type =  input_list[0]
    if radio_type.lower()=='all':
        ret = well_type_options
    elif radio_type.lower()=='productive':
        ret = [wto for wto in well_type_options if wto['value'] in ["GD", "GE", "GW", "IG", "IW", "OD", "OE", "OW"]]
    else:
        ret = {}
    return [ret]
link_radio2_dropdown2 = DashLink([(radio_type,'value')],[(dropdown_type,'options')],build_link_radio2_dropdown2)
dap.adl_append(link_radio2_dropdown2)
link_radio2_dropdown2_value = DashLink(
    [(radio_type,'value')],
    [(dropdown_type,'value')],
    lambda input_list:[[wto['value'] for wto in build_link_radio2_dropdown2(input_list)[0]]])
dap.adl_append(link_radio2_dropdown2_value)

r2c1 = dgr('r2c1',[slider_div,radio1,dropdown1,radio2,dropdown2],child_class=pnnc)


# ### Build the store that reacts to controls in r2c1, and creates:
# 1. ```r2c2_store``` a dcc.Store component to hold well data and aggregates
# 2. ```r2c2c1``` show production volumes by well type panels
# 3. ```r2c2r2``` show production volumes by year graph

# ### ```r2c2_store``` a dcc.Store component to hold well data and aggregates

# In[16]:


r2c2_store = dcc.Store(id='r2c2_store',data={})
def _build_df_from_input_list(input_list,logger=None):
    year_list = input_list[0]
    year_low = int(str(year_list[0]))
    year_high = int(str(year_list[1]))
    df_temp = df_well_info[(df_well_info.Year_Well_Completed>=year_low) & (df_well_info.Year_Well_Completed<=year_high)]
    status_list = input_list[1] 
    type_list = input_list[2]
    try:
        df_temp = df_temp[df_temp.Well_Status.isin(status_list)]
        df_temp = df_temp[df_temp.Well_Type.isin(type_list)]
    except Exception as e:
        if logger is not None:
            logger.warn(f'EXCEPTION: _build_main_data_dictionary {str(e)}')
    return df_temp

def get_well_aggregates(df_in,year_array,well_status_list=None,well_type_list=None):
    df_temp = df_in[(df_in.Year_Well_Completed>=year_array[0]) & (df_in.Year_Well_Completed<=year_array[1])]
    df_ptemp = df_well_production_by_year[(df_well_production_by_year.year>=year_array[0])&(df_well_production_by_year.year<=year_array[1])]
    df_ptemp = df_ptemp[df_ptemp.API_WellNo.isin(df_temp.API_WellNo.unique())]
    if well_status_list is not None:
        valid_status_ids = df1[df1.Well_Status.isin(well_status_list)].API_WellNo.values
        df_ptemp = df_ptemp[df_ptemp.API_WellNo.isin(valid_status_ids)]
    if well_type_list is not None:
        valid_type_ids = df1[df1.Well_Type.isin(well_type_list)].API_WellNo.values
        df_ptemp = df_ptemp[df_ptemp.API_WellNo.isin(valid_type_ids)]
    wells = len(df_temp.API_WellNo.unique())
    agg_data =  {
        'wells':wells,
        'gas':round(df_ptemp.gas.sum()/1000000,1),
        'water':round(df_ptemp.water.sum()/1000000,1),
        'oil':round(df_ptemp.oil.sum()/1000000,1)
    }
    return agg_data

def build_link_r2c2_store(input_list):
    year_list = input_list[0]
    year_low = int(str(year_list[0]))
    year_high = int(str(year_list[1]))
    df_temp = _build_df_from_input_list(input_list,logger)
    df_temp2 = df_temp[["API_WellNo", "Year_Well_Completed"]].groupby(['Year_Well_Completed'],as_index=False).count()
    #{'wells': 14628, 'gas': 865.4, 'water': 15.8, 'oil': 3.8}
    aggs = get_well_aggregates(df_temp,[year_low,year_high])
    ret = {
        'data':df_temp2.to_dict('rows'),
        'no_wells':f"{aggs['wells']} No of",
        'gas_mcf':f"{aggs['gas']}mcf",
        'oil_bbl':f"{aggs['oil']}M bbl",
        'water_bbl':f"{aggs['water']}M bbl"
    }
    return [ret]
link_r2c2_store = DashLink(
    [(slider,'value'),(dropdown_status,'value'),(dropdown_type,'value')],
    [(r2c2_store,'data')],build_link_r2c2_store)
dap.adl_append(link_r2c2_store)


# ### Row 2 Column 2 Row 1
# ##### ```r2c2r1``` show production volumes by well type panels

# In[17]:


# build the panels
no_wells = html.Div(id='no_wells')
gas_mcf = html.Div(id='gas_mcf')
oil_bbl = html.Div(id='oil_bbl')
water_bbl = html.Div(id='water_bbl')
r2c2r1_panel_ids = ['no_wells','gas_mcf','oil_bbl','water_bbl']

def build_panel_link_closure(panel_id):
    def build_panel_link(input_list):
        r2c2_store = input_list[0]
        if panel_id not in r2c2_store:
            stop_callback(f'build_panel_link no data for panel_id {panel_id} with data {input_list}',logger)        
        ret = r2c2_store[panel_id]
        return [ret]
    return build_panel_link
    
for panel_id in r2c2r1_panel_ids:
    r2c2r1_link = DashLink(
        [(r2c2_store,'data')],[(panel_id,'children')],
        build_panel_link_closure(panel_id)
    ) 
    dap.adl_append(r2c2r1_link)

wells_div = html.Div([no_wells,html.P('Wells')],id='wells_div')
gas_div = html.Div([gas_mcf,html.P('Gas')],id='gas_div')
oil_div = html.Div([oil_bbl,html.P('Oil')],id='oil_div')
water_div = html.Div([water_bbl,html.P('Water')],id='water_div')
r2c2r1 = dgr('r2c2r1',[wells_div,gas_div,oil_div,water_div],child_class=pn,parent_class=pnnc)


# ### Row 2 - Column 2 - Row 2
# #### ```r2c2r2``` show production volumes by year graph

# In[18]:



def xygraph_fig(df_wells_per_year):
    xyfig = plotly_plot(
        df_in=df_wells_per_year,
        x_column='Year_Well_Completed',bar_plot=True,number_of_ticks_display=10,
        plot_title='Completed Wells/Year')
    return xyfig

df_wells_per_year = df_well_info[["API_WellNo", "Year_Well_Completed"]].groupby(['Year_Well_Completed'],as_index=False).count()
init_fig = xygraph_fig(df_wells_per_year)
xygraph = dcc.Graph(id='xygraph',figure=init_fig)

def build_link_store_xygraph(input_list):
    r2c2_store = input_list[0]
    if 'data' not in r2c2_store:
        stop_callback(f'build_link_store_xygraph no data {input_list}',logger)        
    dict_df = r2c2_store['data']
    df_wells_per_year = make_df(dict_df)
    xyfig = xygraph_fig(df_wells_per_year)
    return [xyfig]
link_store_xygraph = DashLink([(r2c2_store,'data')],[(xygraph,'figure')],build_link_store_xygraph)
dap.adl_append(link_store_xygraph)

r2c2r2 = dgr('r2c2r2',[xygraph])


# ### Row 2 - combine the above to create row 2

# In[19]:


r2c2r3 = html.Div()
r2c2 = dgr('r2c2',[r2c2r1,r2c2r2,r2c2r3,r2c2_store],child_class=None,parent_class=pnnc)
r2 = dgr('r2',[r2c1,r2c2])


# ### Row 3
# #### ```r3``` Create a sattelite map and a pie chart of production

# In[20]:


# ************* Build row 3: the map and pie charts ********************************
mapbox_access_token = "pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w"

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)

def _create_map_from_df(df_in):
    traces = []
    for wtype, dfff in df_in.groupby("wtype"):
        trace = dict(
            type="scattermapbox",
            lon=dfff["Surface_Longitude"],
            lat=dfff["Surface_latitude"],
            text=dfff["Well_Name"],
            customdata=dfff["API_WellNo"],
            name=wtype,
            marker=dict(size=4, opacity=0.6),
        )
        traces.append(trace)

    figure = dict(data=traces, layout=layout)
    return figure
    

def build_map_figure(input_list):
    dff = _build_df_from_input_list(input_list)
    fig = _create_map_from_df(dff)
    return [fig]

def _create_pie_figure_from_df(dff,years):
    layout_pie = copy.deepcopy(layout)
    selected = dff["API_WellNo"].values
    
    year_low = years[0]
    year_high = years[1]
    aggs = get_well_aggregates(dff,[year_low,year_high])

    index = aggs['wells']
    gas = aggs['gas']
    oil = aggs['oil']
    water = aggs['water']
    
    # create a DataFrame that holds counts of wells by type, sorted by wtype
    df_well_count = dff[['wtype','wcolor']].sort_values("wtype").groupby(["wtype"],as_index=False).count()
    df_well_count = df_well_count.rename(columns={'wcolor':'type_count'})

    # Create a DataFrame that holds unique combinations of type and color, sorted by wtype
    df_well_color = dff[['wtype','wcolor']].drop_duplicates().sort_values('wtype')    
    
    data = [
        dict(
            type="pie",
            labels=["Gas", "Oil", "Water"],
            values=[gas, oil, water],
            name="Production Breakdown",
            text=[
                "Total Gas Produced (mcf)",
                "Total Oil Produced (bbl)",
                "Total Water Produced (bbl)",
            ],
            hoverinfo="text+value+percent",
            textinfo="label+percent+name",
            hole=0.5,
            marker=dict(colors=["#fac1b7", "#a9bb95", "#92d8d8"]),
            domain={"x": [0, 0.45], "y": [0.2, 0.8]},
        ),
        dict(
            type="pie",
            labels=df_well_count.wtype.values,  # this works b/c it's sorted by wtype
            values=df_well_count.type_count.values, # this works b/c it's sorted by wtype
            name="Well Type Breakdown",
            hoverinfo="label+text+value+percent",
            textinfo="label+percent+name",
            hole=0.5,
            marker=dict(colors=df_well_color.wcolor.values), # this works b/c it's sorted by wtype
            domain={"x": [0.55, 1], "y": [0.2, 0.8]},
        ),
    ]
    layout_pie["title"] = "Production Summary: {} to {}".format(
        year_low, year_high
    )
    layout_pie["font"] = dict(color="#777777")
    layout_pie["legend"] = dict(
        font=dict(color="#CCCCCC", size="10"), orientation="h", bgcolor="rgba(0,0,0,0)"
    )

    figure = dict(data=data, layout=layout_pie)
    return figure

def build_pie_figure(input_list):
    year_list = input_list[0]
    dff = _build_df_from_input_list(input_list)
    fig = _create_pie_figure_from_df(dff,year_list)
    return [fig]

# mapgraph = dgc.FigureComponent('mapgraph',None,_create_map_figure,input_tuple_list=input_component_list)
init_map_figure = _create_map_from_df(df_well_info)
mapgraph = dcc.Graph(id='mapgraph',figure=init_map_figure)
link_mapgraph = DashLink(
    [(slider,'value'),(dropdown_status,'value'),(dropdown_type,'value')],
    [(mapgraph,'figure')],build_map_figure)
dap.adl_append(link_mapgraph)

# piegraph = dgc.FigureComponent('piegraph',None,_create_pie_figure,input_tuple_list=input_component_list)
init_years = [df_well_info.Year_Well_Completed.min(), df_well_info.Year_Well_Completed.max()]
init_pie_figure = _create_pie_figure_from_df(df_well_info,init_years)
piegraph = dcc.Graph(id='piegraph',figure=init_pie_figure)
link_piegraph = DashLink(
    [(slider,'value'),(dropdown_status,'value'),(dropdown_type,'value')],
    [(piegraph,'figure')],build_pie_figure)
dap.adl_append(link_piegraph)



# ### Put all rows together and build the app

# In[21]:


rside = html.Div(' ')
r3 = dgr('r3',[mapgraph,html.Div(' '),piegraph],parent_class=pnnc)
rbot = dgr('rbot',['the bottom'],child_class=pn)
rall_rows = dgr('rall_rows',[r1,r2,r3,rbot])
rall_cols = dgr('rall_cols',[rside,rall_rows,rside],parent_class=pnnc)
if __name__=='__main__':
    app_host = '127.0.0.1'
    app_port = 8050
    if url_base_pathname is not None:
        app = dash.Dash(__name__, url_base_pathname=url_base_pathname)
    else:
        app = dash.Dash(__name__)
    app.title = 'Oil and Gas'    
    app.layout = html.Div([rall_cols])
    for dl in dap.all_dash_links:
        dl.callback(app)
    full_url = f"http://{app_host}:{app_port}" + url_base_pathname
    logger.info(f"This app will run at the URL: {full_url}")
    app.run_server(host=app_host,port=app_port)


# In[ ]:





# In[22]:


# !jupyter nbconvert --to script dash_oil_gas.ipynb


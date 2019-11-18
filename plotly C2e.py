import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.io as pio

df=pd.read_csv("D://Tree Data C2E clean.csv")
melted=df.melt(id_vars=['Tree ID','Tree Species'],value_vars=['4/29/2016','10/20/2016','4/10/2017','9/11/2017'],
               var_name='Date',value_name='Height')
species=df.groupby('Tree Species').mean()
species=species.reset_index()
species=species.drop('Tree ID',axis=1)
sp_melt=species.melt(id_vars=['Tree Species'],value_vars=['4/29/2016','10/20/2016','4/10/2017','9/11/2017'],
                     var_name='Date',value_name='Height')
sp_melt['Date']=pd.to_datetime(sp_melt['Date'])
stdate=sp_melt['Date'].iloc[0]
sp_melt['Days']=(sp_melt['Date']-stdate).dt.days
sp_list=np.unique(sp_melt['Tree Species'])
sp_list=[sp for sp in sp_list if sp!='Screwbean Mesquite/Ironwood']
def sp_frame(sp):
    return sp_melt[sp_melt['Tree Species']==sp]
def regress(spdf):
    X=spdf['Days'].values.reshape(-1,1)
    y=spdf['Height'].values.reshape(-1,1)
    reg=LinearRegression()
    reg.fit(X,y)
    return reg
sp_regs={sp:{'frame':sp_frame(sp),'reg':regress(sp_frame(sp))} for sp in sp_list}
fig=px.scatter(sp_melt,x='Days',y='Height',color='Tree Species',trendline='ols')
fig.write_image("fig2.jpeg")
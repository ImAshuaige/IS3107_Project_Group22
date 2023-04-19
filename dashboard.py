from dash import Dash, html, dcc, Input, Output
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
import pmdarima as pm

from sklearn.metrics import r2_score

import dash_bootstrap_components as dbc

#connection to the database

#replace password for database
password = "password"

#replace database schema name
db_schema = "IS3107_Project"

alchemyEngine = create_engine(
    "postgresql+psycopg2://postgres:" + password + "@localhost:5432/" + db_schema
)
dbConnection = alchemyEngine.connect()

# date price
df = pd.read_sql("select * from public.bitcoin_prices", dbConnection)
pd.set_option("display.expand_frame_repr", False)

df1 = df[{"price", "date"}]  # date price only

price_df = pd.read_sql("select * from public.bitcoin_prices", dbConnection);
price_dataset = price_df[{"date", "price"}]
price_dataset["date"] = pd.to_datetime(price_dataset["date"])
price_dataset = price_dataset.groupby("date").sum()

sentiment_selected_cols = {
    "date",
    "negative",
    "positive",
    "neutral",
    "compound",
}
sentiment_df = pd.read_sql(
    "SELECT * FROM public.bitcoin_tweets_sentiment",
    dbConnection,
)
sentiment_dataset = sentiment_df[sentiment_selected_cols] 
sentiment_dataset["date"] = pd.to_datetime(sentiment_dataset["date"])
sentiment_dataset = sentiment_dataset.groupby("date").sum()

combined_dataset = price_dataset.merge(sentiment_dataset, on="date")

X = combined_dataset[["negative", "positive", "neutral", "compound"]]
y = combined_dataset["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy_LR = r2_score(y_test, y_pred)

df1 = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
df1.reset_index(inplace=True)
df1.sort_values(by="date", inplace=True)

# Predicting tomorrow's price
presentday = datetime.today()
tomorrow = presentday + timedelta(1)
data = {
    "negative" : combined_dataset.iloc[len(combined_dataset)-1].negative,
    "positive" : combined_dataset.iloc[len(combined_dataset)-1].positive,
    "neutral" : combined_dataset.iloc[len(combined_dataset)-1].neutral,
    "compound" : combined_dataset.iloc[len(combined_dataset)-1].compound,
}
pred = pd.DataFrame(data, index=[0])
final_pred = model.predict(pred)
y_pred = final_pred[0]
pred['Predicted'] = y_pred

line1 = go.Scatter(
    x=df1["date"],
    y=df1["Actual"],
    mode="lines",
    marker_color=px.colors.qualitative.Dark24[1],
    name="Actual",
)
line2 = go.Scatter(
    x=df1["date"],
    y=df1["Predicted"],
    mode="lines",
    marker_color=px.colors.qualitative.Dark24[0],
    name="Predicted",
)

data1 = [line1, line2]
layout1 = go.Layout(title="Linear Regression with Tweet Sentiment", xaxis=dict(tickformat="%Y-%m-%d"))
fig = go.Figure(data=data1, layout=layout1)
fig.update_layout(
    title=dict(x=0.5),
    xaxis_title="Date",
    yaxis_title="Price",
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis=dict(tickformat="%Y-%m-%d"),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig.update_xaxes(tickformat="%Y-%m-%d", tickangle=45)
fig.update_yaxes(gridcolor="lightgray")

#see if price is rising or falling
if pred.iloc[0].Predicted > df1.iloc[len(df1) - 1].Actual:
    increaseLR = True
else:
    increaseLR = False

# sentimentprice_linear_regression

price_selected_columns = {"date", "price"}
price_df = pd.read_sql("select * from public.bitcoin_prices", dbConnection)
price_dataset= price_df[price_selected_columns]

price_dataset["date"] = pd.to_datetime(price_dataset["date"])
price_dataset = price_dataset.groupby("date").sum()

sentiment_selected_cols = {
    "date",
    "negative",
    "positive",
    "neutral",
    "compound",
}
sentiment_df = pd.read_sql(
    "SELECT * FROM public.bitcoin_tweets_sentiment",
    dbConnection,
)
sentiment_dataset = sentiment_df[sentiment_selected_cols]
sentiment_dataset["date"] = pd.to_datetime(sentiment_dataset["date"])
sentiment_dataset = sentiment_dataset.groupby("date").sum()

combined_dataset = price_dataset.merge(sentiment_dataset, on="date")

X = combined_dataset[["negative", "positive", "neutral", "compound"]]
y = combined_dataset["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
df.reset_index(inplace=True)
df.sort_values(by="date", inplace=True)

fig2 = go.Figure(
    data=[
        go.Scatter(
            name="Negative", x=combined_dataset.index, y=combined_dataset["negative"]
        ),
        go.Scatter(
            name="Positive", x=combined_dataset.index, y=combined_dataset["positive"]
        ),
        go.Scatter(
            name="Neutral", x=combined_dataset.index, y=combined_dataset["neutral"]
        ),
    ],
    layout=go.Layout(
        title="Sentiment Scores Over Time",
        xaxis=dict(tickformat="%Y-%m-%d"),
        yaxis=dict(title="Total Sentiment Score Per Day"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    ),
)
fig2.update_xaxes(title="Date")
fig2.update_xaxes(tickformat="%Y-%m-%d", tickangle=45)

#time-regression model
price_cols = ["date", "price"]

price_dataset = pd.read_sql(
    "SELECT {} FROM public.bitcoin_prices".format(
        ", ".join(price_cols)
    ), dbConnection,
)
# format price data
price_dataset["date"] = pd.to_datetime(
    price_dataset["date"], format='%Y/%m/%d')
data = price_dataset.dropna()
data = data.set_index('date')
data = data.asfreq('d')

# split into training and testing set
combined_dataset = data.merge(sentiment_dataset, on="date")

y_train, y_test = train_test_split(
    combined_dataset.price, test_size=0.2, random_state=42
)

history = [x for x in y_train]
predictions = list()
# walk-forward validation
for t in range(len(y_test)):
   model = pm.auto_arima(history, trace=True, error_action="ignore", seasonal=True, stepwise=True, suppress_warnings=True)
   model_fit = model.fit(history)
   prediction, confint = model_fit.predict(n_periods=1, return_conf_int=True)
   yhat = prediction[0]
   predictions.append(yhat)
   obs = y_test[t]
   history.append(obs)

result_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
result_df.reset_index(inplace=True)
result_df.sort_values(by="date", inplace=True)

yhat = model.predict(n_periods=1, return_conf_int=False)

# create fig
line1 = go.Scatter(
    x=result_df["date"],
    y=result_df["Actual"],
    mode="lines",
    marker_color=px.colors.qualitative.Dark24[1],
    name="Actual",
)

line2 = go.Scatter(
    x=result_df["date"],
    y=result_df["Predicted"],
    mode="lines",
    marker_color=px.colors.qualitative.Dark24[0],
    name="Predicted",
)
data3 = [line1, line2]
layout3 = go.Layout(title="Price Predicted by Time Regression", xaxis=dict(tickformat="%Y-%m-%d"))
fig3 = go.Figure(data=data3, layout=layout3)
fig3.update_xaxes(tickformat="%Y-%m-%d", tickangle=45)
fig3.update_yaxes(gridcolor="lightgray")
fig3.update_layout(
    title=dict(x=0.5),  # center the title
    xaxis_title="Date",  # setup the x-axis title
    yaxis_title="Price",  # setup the x-axis title
    margin=dict(l=20, r=20, t=60, b=20),  # setup the margin
    xaxis=dict(tickformat="%Y-%m-%d"),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

#see if price is rising or falling
if yhat > result_df.iloc[len(result_df) - 2].Actual:
    increaseTime = True
else:
    increaseTime = False

yhat1, confint = model_fit.predict(n_periods=1, return_conf_int=True)
yhat1 = yhat1[0]

#Accuracy
accuracy_TS = r2_score(y_test, predictions)

#Auto-ARIMA model
price_cols = ["date", "price"]

price_dataset = pd.read_sql(
    "SELECT {} FROM public.bitcoin_prices".format(
        ", ".join(price_cols)
    ), dbConnection,
)

# format price data
price_dataset["date"] = pd.to_datetime(
    price_dataset["date"], format='%Y/%m/%d')
price_dataset = price_dataset.groupby("date").sum()
price_dataset = price_dataset.dropna()
price_dataset = price_dataset.asfreq('d')

sentiment_selected_cols = [
    "date",
    "tweet_id",
    "compound"
]

sentiment_dataset = pd.read_sql(
    "SELECT {} FROM public.bitcoin_tweets_sentiment".format(
        ", ".join(sentiment_selected_cols)
    ),
    dbConnection,
)

sentiment_dataset["date"] = pd.to_datetime(
    sentiment_dataset["date"], format='%Y/%m/%d')
sentiment_dataset = sentiment_dataset.groupby("date").sum()
combined_dataset = price_dataset.merge(sentiment_dataset, on='date')
combined_dataset.dropna()

combined_dataset.reset_index(inplace=True)
exog = combined_dataset["compound"]
exog = exog.dropna()

exog_train = exog[:85]
exog_test = exog[-20:]
train = combined_dataset.price[:85]
test = combined_dataset[-20:]

history = [x for x in train]
exog_history = [z for z in exog_train]

predictions = list()

for t in range(len(test)):
   obs_exog = exog_test.iloc[t]
   model = pm.auto_arima(history, exogenous=exog_history, start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, D=1, start_Q=0, max_P=5, max_D=5, max_Q=5, m=12, seasonal=True,
                            error_action='warn', trace=True, supress_warnings=True, stepwise=True, random_state=20, n_fits=50)
   prediction, confint = model.predict(n_periods=1, X=obs_exog, return_conf_int=True)
   yhat = prediction[0]
   predictions.append(yhat)
   exog_history.append(obs_exog)
   obs = test.price.iloc[t]
   history.append(obs)
   
result_df = pd.DataFrame({"date": test.date, "Actual": test.price, "Predicted": predictions})
result_df.reset_index(inplace=True)
result_df.sort_values(by="date", inplace=True)

arimaExog_accuracy = r2_score(result_df.Actual, result_df.Predicted)

#predict values for tomorrow based on last set of values in table
obs_exog = exog.iloc[len(exog) -1]
yhat2 = model.predict(n_periods=1, X=obs_exog, return_conf_int=False)
price_dataset = price_dataset.reset_index(drop=False, inplace=False)

# create fig
line1 = go.Scatter(
    x=result_df["date"],
    y=result_df["Actual"],
    mode="lines",
    marker_color=px.colors.qualitative.Dark24[1],
    name="Actual",
)
line2 = go.Scatter(
    x=result_df["date"],
    y=result_df["Predicted"],
    mode="lines",
    marker_color=px.colors.qualitative.Dark24[0],
    name="Predicted",
)
data4 = [line1, line2]
layout4 = go.Layout(title="Price Predicted by ARIMA model with exogenous variables", xaxis=dict(tickformat="%Y-%m-%d"))
fig4 = go.Figure(data=data4, layout=layout4)
fig4.update_xaxes(tickformat="%Y-%m-%d", tickangle=45)
fig4.update_yaxes(gridcolor="lightgray")
fig4.update_layout(
    title=dict(x=0.5),  # center the title
    xaxis_title="Date",  # setup the x-axis title
    yaxis_title="Price",  # setup the x-axis title
    margin=dict(l=20, r=20, t=60, b=20),  # setup the margin
    xaxis=dict(tickformat="%Y-%m-%d"),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

#see if price is rising or falling
if yhat2 > result_df.iloc[len(result_df) - 2].Actual:
    increaseARIMA = True
else:
    increaseARIMA = False

# RandomForestRegressor
price_selected_columns = {"date", "price"}
price_df = pd.read_sql("select * from public.bitcoin_prices", dbConnection)
price_dataset= price_df[price_selected_columns]

price_dataset["date"] = pd.to_datetime(price_dataset["date"])
price_dataset = price_dataset.groupby("date").sum()

sentiment_selected_cols = {
    "date",
    "compound",
}
sentiment_df = pd.read_sql(
    "SELECT * FROM public.bitcoin_tweets_sentiment",
    dbConnection,
)
sentiment_dataset = sentiment_df[sentiment_selected_cols]
sentiment_dataset["date"] = pd.to_datetime(sentiment_dataset["date"])
sentiment_dataset = sentiment_dataset.groupby("date").mean()

combined_dataset = price_dataset.merge(sentiment_dataset, on="date")

X = combined_dataset[["compound"]]
y = combined_dataset["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)
y_pred2 = model.predict(X_test)

result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred2})
result_df.reset_index(inplace=True)
result_df.sort_values(by="date", inplace=True)

#Predicting tomorrow's price
presentday = datetime.today()
tomorrow = presentday + timedelta(1)
data = {
          "date": [tomorrow.strftime("%Y-%m-%d")],
        }
pred = pd.DataFrame(data)
pred['date'] = pd.to_datetime(pred['date'])
pred['date'] = pred['date'].map(dt.datetime.toordinal)
        
pred = pred['date'].values.reshape(-1, 1)
final_pred = model.predict(pred)
rfr_pred = final_pred[0]

#see if price is rising or falling compared to yesterday's actual price
if rfr_pred > result_df.iloc[len(result_df) - 1].Actual:
    increaseRFR = True
else:
    increaseRFR= False
    
#calculate accuracy score
accuracy_RFR = r2_score(y_test, y_pred2)
    
# create fig
line1 = go.Scatter(
    x=result_df["date"],
    y=result_df["Actual"],
    mode="lines",
    marker_color=px.colors.qualitative.Dark24[1],
    name="Actual",
)
line2 = go.Scatter(
    x=result_df["date"],
    y=result_df["Predicted"],
    mode="lines",
    marker_color=px.colors.qualitative.Dark24[0],
    name="Predicted",
)

data5 = [line1, line2]
layout5 = go.Layout(title="Price Predicted by Daily Average Compound Sentiment Score", xaxis=dict(tickformat="%Y-%m-%d"))
fig5 = go.Figure(data=data5, layout=layout5)
fig5.update_layout(
    title=dict(x=0.5),
    xaxis_title="Date",
    yaxis_title="Price",
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis=dict(tickformat="%Y-%m-%d"),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig5.update_xaxes(tickformat="%Y-%m-%d", tickangle=45)
fig5.update_yaxes(gridcolor="lightgray")


# building app

app = Dash(
    external_stylesheets = [dbc.themes.BOOTSTRAP]
)

colors = {"bg": "#fce0e0", "h1_bg": "white", "background": "#f7cf68", "text": "#423636"}

app.layout = html.Div(
    style={"background": colors["bg"]},
    children=[
        html.H1(
            children="How do Tweets affect Bitcoin prices?",
            style={
                "textAlign": "center",
                "background": colors["bg"],
                "color": colors["text"],
                "height": "50px",
                "padding":"3vh",
            },
        ),

        html.Div(
            style={
                "padding" : "20px"
            },
            children=[
                html.H4(
                    children="Our predictions for today",
                    style={
                        "textAlign": "center",
                        "background": colors["bg"],
                        "color": colors["text"],
                        "height": "50px",
                        "padding":"3vh",
                    },
                ),

                dbc.Row(
                    style = {
                        "margin": "auto"
                    },
                    children = [
                        dbc.Col(
                            dbc.Card(
                                [
                                dbc.CardBody(
                                        [
                                            html.H4("Time Regression", className="card-title"),
                                            html.P(
                                                "The predicted price for tomorrow based on time regression is: ",
                                                className="card-text",
                                            ),
                                            html.H4(
                                                "{:.8f}".format(yhat1),
                                                className="card-text text-success" if increaseTime else "card-text text-danger",
                                            )
                                        ]
                                    ),
                                ],
                                className="shadow",
                                style={
                                    "margin": "2vh",
                                }
                            ),
                            width=3
                        ),
                         dbc.Col(
                            dbc.Card(
                                [
                                dbc.CardBody(
                                        [
                                            html.H4("ARIMA model", className="card-title"),
                                            html.P(
                                                "The predicted price for tomorrow based on ARIMA model with exogenous variables is: ",
                                                className="card-text",
                                            ),
                                            html.H4(
                                                "{:.8f}".format(yhat2[0]),
                                                className="card-text text-success" if increaseARIMA else "card-text text-danger",
                                            )
                                        ]
                                    ),
                                ],
                                className="shadow",
                                style={
                                    "margin": "2vh",
                                }
                            ),
                            width=3
                        ),

                        dbc.Col(
                            dbc.Card(
                                [
                                dbc.CardBody(
                                        [
                                            html.H4("Random Forest Regressor", className="card-title"),
                                            html.P(
                                                "The predicted price for tomorrow based on Random Forest Regressor is: ",
                                                className="card-text",
                                            ),
                                            html.H4(
                                                str("%.8f" % round(rfr_pred,8)),
                                                className="card-text text-success" if increaseRFR else "card-text text-danger",
                                            )
                                        ]
                                    ),
                                ],
                                className="shadow",
                                style={
                                    "margin": "2vh",
                                }
                            ),
                            width=3
                        ),
                        
                        dbc.Col(
                            dbc.Card(
                                [
                                dbc.CardBody(
                                        [
                                            html.H4("Linear Regression with Bitcoin Prices", className="card-title"),
                                            html.P(
                                                "The predicted price for tomorrow based on Linear Regression is: ",
                                                className="card-text",
                                            ),
                                            html.H4(
                                                "{:.8f}".format(y_pred[0]),
                                                className="card-text text-success" if increaseLR
                                                 else "card-text text-danger",
                                            )
                                        ]
                                    ),
                                ],
                                className="shadow",
                                style={
                                    "margin": "2vh",
                                }
                            ),
                            width=3
                        ),
                    ]
                )
    
            ]
        ),

        html.Div(
            style = {
                "padding-top": "20px",
            },
            children = [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                dbc.CardBody(
                                        [
                                            html.H2(
                                                    children="Random Forest Model with Tweet Sentiment",
                                                    style={
                                                        "textAlign": "center",
                                                        "backgroundColor": colors["background"],
                                                        "height": "55px",
                                                        "padding": "10px",
                                                        "border-radius": "15px",
                                                        "color": "white",
                                                    },
                                                ),
                                            dcc.Graph(figure=fig5, id="time_regression"),
                                            
                                            html.H5(
                                                children="R-squared Score for Random Forest Model: " +  str(accuracy_LR),
                                                style={
                                                    "textAlign": "center",
                                                    "color": "black",
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                                className="shadow",
                                style={
                                    "margin": "2vh",
                                }
                            ),
                            width=6
                        ),
                        
                        dbc.Col(
                            dbc.Card(
                                [
                                dbc.CardBody(
                                        [
                                            html.H2(
                                                children="ARIMA model based on historical prices",
                                                style={
                                                    "textAlign": "center",
                                                    "backgroundColor": colors["background"],
                                                    "height": "55px",
                                                    "padding": "10px",
                                                    "border-radius": "15px",
                                                    "color": "white",
                                                },
                                            ),
                                            dcc.Graph(figure=fig3, id="time-regression"),
                                            
                                            html.H5(
                                                children="R-squared Score for Random Forest Model: " +  str(accuracy_TS),
                                                style={
                                                    "textAlign": "center",
                                                    "color": "black",
                                                },
                                            ),
                                            
                                        ]
                                    ),
                                ],
                                className="shadow",
                                style={
                                    "margin": "2vh",
                                }
                            ),
                            width=6
                        ),
                        
                    ]
                ),
            ],
        ),

        html.Div(
            style = {
                "padding-top": "20px",
            },
            children = [
                dbc.Row( 
                    [
                        dbc.Col(
                            dbc.Card(
                                [
                                dbc.CardBody(
                                        [
                                            html.H2(
                                                children="Linear Regression with Tweet Sentiment",
                                                style={
                                                    "textAlign": "center",
                                                    "backgroundColor": colors["background"],
                                                    "height": "55px",
                                                    "padding": "10px",
                                                    "border-radius": "15px",
                                                    "color": "white",
                                                },
                                            ),
                                            dcc.Graph(figure=fig, id="linear-regression"),
                                            
                                            html.H5(
                                                children="R-Squared Score for Linear Regression: " +  str(accuracy_LR),
                                                style={
                                                    "textAlign": "center",
                                                    "color": "black",
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                                className="shadow",
                                style={
                                    "margin": "2vh",
                                }
                            ),
                            width=6
                        ),
                        
                        dbc.Col(
                            dbc.Card(
                                [
                                dbc.CardBody(
                                        [
                                            html.H2(
                                                children="ARIMA Model with Tweet Sentiment",
                                                style={
                                                    "textAlign": "center",
                                                    "backgroundColor": colors["background"],
                                                    "height": "55px",
                                                    "padding": "10px",
                                                    "border-radius": "15px",
                                                    "color": "white",
                                                },
                                            ),
                                            dcc.Graph(figure=fig4, id="arima-model"),
                                            html.H5(
                                                children="R-squared score for ARIMA model with exogenous variables: " +  str(arimaExog_accuracy),
                                                style={
                                                    "textAlign": "center",
                                                    "color": "black",
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                                className="shadow",
                                style={
                                    "margin": "2vh",
                                }
                            ),
                            width=6
                        ),
                    ]
                ),
            ],
        )
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True, port=8051)  # http://127.0.0.1:8051

#!/usr/bin/env python3
from flask import Flask, request, render_template, session, redirect
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from tqdm import tqdm

app = Flask(__name__)

dftlg = pd.read_csv('static/csv/random1000tlg.csv', sep=',')

dftwt = pd.read_csv('static/csv/random1000twt.csv', sep=',')

dfinv = pd.read_csv('static/csv/random1000inv.csv', sep=',')

@app.route('/', methods=("POST", "GET"))
def home():
    return render_template('home.html', zip=zip)

@app.route('/hometwt.html', methods=("POST", "GET"))
def hometwt():
    return render_template('hometwt.html', column_names=dftwt.columns.values, row_data=list(dftwt.values.tolist()), link_column="username", zip=zip)

@app.route('/hometlg.html', methods=("POST", "GET"))
def hometlg():
    return render_template('hometlg.html', column_names=dftlg.columns.values, row_data=list(dftlg.values.tolist()), link_column="user_id", zip=zip)

@app.route('/homeinv.html', methods=("POST", "GET"))
def homeinv():
    return render_template('homeinv.html', column_names=dfinv.columns.values, row_data=list(dfinv.values.tolist()), link_column="user_id", zip=zip)

@app.route('/usertwt.html')
def usertwt():
    df_monthly = pd.read_csv("static/csv/aslanamcaaylık.csv")

    username = request.args.get('username')
    user_data = df_monthly[df_monthly['username'] == username]
    user_data = user_data.drop('username', axis=1)

    fig, ax = plt.subplots()
    ax.plot(df_monthly.message_date.str[:10], df_monthly.stock_change)
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Manipülasyon Yüzdesi')
    ax.set_title('@' + username + ' Borsa Manipülasyonu')
    ax.tick_params(axis='x', rotation=90)
    fig.subplots_adjust(bottom=0.2)

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    image_data = base64.b64encode(buffer.getvalue()).decode()

    return render_template('usertwt.html', username=username, column_names=user_data.columns.values, row_data=list(user_data.values.tolist()), zip=zip, image_data=image_data)

@app.route('/labeltwt.html', methods=("POST", "GET"))
def labeltwt():
    return render_template('labeltwt.html', column_names=dftwt.columns.values, row_data=list(dftwt.values.tolist()), link_column="username", zip=zip)

@app.route('/labeltlg.html', methods=("POST", "GET"))
def labeltlg():
    return render_template('labeltlg.html', column_names=dftlg.columns.values, row_data=list(dftlg.values.tolist()), zip=zip)

@app.route('/labelinv.html', methods=("POST", "GET"))
def labelinv():
    return render_template('labelinv.html', column_names=dfinv.columns.values, row_data=list(dfinv.values.tolist()), zip=zip)

@app.route('/fileuploadtwt.html', methods=("POST", "GET"))
def fileuploadtwt():
    return render_template('fileuploadtwt.html', column_names=dfinv.columns.values, row_data=list(dfinv.values.tolist()), zip=zip)

@app.route('/fileuploadtlg.html', methods=("POST", "GET"))
def fileuploadtlg():
    return render_template('fileuploadtlg.html', column_names=dfinv.columns.values, row_data=list(dfinv.values.tolist()), zip=zip)

@app.route('/fileuploadinv.html', methods=("POST", "GET"))
def fileuploadinv():
    return render_template('fileuploadinv.html', column_names=dfinv.columns.values, row_data=list(dfinv.values.tolist()), zip=zip)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
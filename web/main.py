#!/usr/bin/env python3
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from flask import Flask, request, render_template
import pandas as pd
import csv
import os
import re

app = Flask(__name__)

# Load the models once
device = torch.device("cpu") # Use CPU device
model = BertForSequenceClassification.from_pretrained(
    'dbmdz/bert-base-turkish-cased', num_labels=3)
model.load_state_dict(torch.load("static/pth/model.pth", map_location=device))
model.to(device)

# Load the BERT tokenizer and tokenize the input texts
tokenizer = BertTokenizer.from_pretrained(
    'dbmdz/bert-base-turkish-cased')

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route('/', methods=("POST", "GET"))
def home():
    return render_template('home.html', zip=zip)


@app.route('/hometwt.html', methods=("POST", "GET"))
def hometwt():
    predicttwt()

    dftwt = pd.read_csv('static/csv/twitter.csv', sep=';')
    return render_template('hometwt.html', column_names=dftwt.columns.values, row_data=list(dftwt.values.tolist()), link_column="username", zip=zip)


@app.route('/hometlg.html', methods=("POST", "GET"))
def hometlg():
    predicttlg()

    dftlg = pd.read_csv('static/csv/telegram.csv', sep=';')
    return render_template('hometlg.html', column_names=dftlg.columns.values, row_data=list(dftlg.values.tolist()), link_column="user_id", zip=zip)


@app.route('/homeinv.html', methods=("POST", "GET"))
def homeinv():
    predictinv()

    dfinv = pd.read_csv('static/csv/investing.csv', sep=';')
    return render_template('homeinv.html', column_names=dfinv.columns.values, row_data=list(dfinv.values.tolist()), link_column="user_id", zip=zip)


@app.route('/labeltwt.html', methods=("POST", "GET"))
def labeltwt():
    dftwt = pd.read_csv('static/csv/twitter.csv', sep=';')
    return render_template('labeltwt.html', column_names=dftwt.columns.values, row_data=list(dftwt.values.tolist()), link_column="username", zip=zip)


@app.route('/labeltlg.html', methods=("POST", "GET"))
def labeltlg():
    dftlg = pd.read_csv('static/csv/telegram.csv', sep=';')
    return render_template('labeltlg.html', column_names=dftlg.columns.values, row_data=list(dftlg.values.tolist()), zip=zip)


@app.route('/labelinv.html', methods=("POST", "GET"))
def labelinv():
    dfinv = pd.read_csv('static/csv/investing.csv', sep=';')
    return render_template('labelinv.html', column_names=dfinv.columns.values, row_data=list(dfinv.values.tolist()), zip=zip)


@app.route('/fileuploadtwt.html', methods=("POST", "GET"))
def fileuploadtwt():
    dftwt = pd.read_csv('static/csv/twitter.csv', sep=';')
    return render_template('fileuploadtwt.html', column_names=dftwt.columns.values, row_data=list(dftwt.values.tolist()), zip=zip)


@app.route('/fileuploadtlg.html', methods=("POST", "GET"))
def fileuploadtlg():
    dftlg = pd.read_csv('static/csv/telegram.csv', sep=';')
    return render_template('fileuploadtlg.html', column_names=dftlg.columns.values, row_data=list(dftlg.values.tolist()), zip=zip)


@app.route('/fileuploadinv.html', methods=("POST", "GET"))
def fileuploadinv():
    dfinv = pd.read_csv('static/csv/investing.csv', sep=';')
    return render_template('fileuploadinv.html', column_names=dfinv.columns.values, row_data=list(dfinv.values.tolist()), zip=zip)

@app.route('/saveFormDatatwt', methods=['POST'])
def saveFormDatatwt():
    # Get the form data
    tweet_id = request.form.get('tweet_id')
    followers = request.form.get('followers')
    date = request.form.get('date')
    message_detail = request.form.get('message_detail')
    retweet_count = request.form.get('retweet_count')
    username = request.form.get('username')

    # Check if the dataset file exists
    csv_file_path = 'static/csv/twitter.csv'
    file_exists = os.path.isfile(csv_file_path)

    # Append the form data to the existing CSV file with specified encoding
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')

        # Add attribute names as the first row if the file doesn't exist
        if not file_exists:
            writer.writerow(['tweet_id', 'followers', 'date',
                            'message_detail', 'retweet_count', 'username', 'label'])

        writer.writerow([tweet_id, followers, date, message_detail, retweet_count, username, ''])

    return 'twt form data added to twitter.csv'

@app.route('/saveFormDatatlg', methods=['POST'])
def saveFormDatatlg():
    # Get the form data
    comment_id = request.form.get('comment_id')
    content = request.form.get('content')
    date = request.form.get('date')
    link = request.form.get('link')
    username = re.search(r'http://t.me/(\w+)/\d+', link).group(1)

    # Check if the dataset file exists
    csv_file_path = 'static/csv/telegram.csv'
    file_exists = os.path.isfile(csv_file_path)

    # Append the form data to the existing CSV file with specified encoding
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')

        # Add attribute names as the first row if the file doesn't exist
        if not file_exists:
            writer.writerow(['comment_id', 'content', 'date',
                            'link', 'username', 'label'])

        writer.writerow([comment_id, content, date, link, username, ''])

    return 'tlg form data added to telegram.csv'


@app.route('/saveFormDatainv', methods=['POST'])
def saveFormDatainv():
    # Get the form data
    comment_id = request.form.get('comment_id')
    content = request.form.get('content')
    date = request.form.get('date')
    link = request.form.get('link')
    user_id = request.form.get('user_id')
    name = request.form.get('name')

    # Check if the dataset file exists
    csv_file_path = 'static/csv/investing.csv'
    file_exists = os.path.isfile(csv_file_path)

    # Append the form data to the existing CSV file with specified encoding
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')

        # Add attribute names as the first row if the file doesn't exist
        if not file_exists:
            writer.writerow(['comment_id', 'content', 'date',
                            'link', 'user_id', 'name', 'label'])

        writer.writerow([comment_id, content, date, link, user_id, name, ''])

    return 'inv form data added to investing.com'

@app.route('/predicttwt', methods=['POST', 'GET'])
def predicttwt():
    # Load the test set and create input tensors
    # Assume that the test data is in CSV format with columns: 'message_detail', 'username', and 'content'
    test_data = pd.read_csv('static/csv/twitter.csv', sep=';', encoding='utf-8')

    # Filter out the rows with 'nan' content
    test_data = test_data.dropna(subset=['message_detail'])

    if len(test_data) > 0:
        # Iterate over each row in the test data
        for i, row in test_data.iterrows():
            # Check if the label is NaN
            if pd.isnull(row['label']):
                # Retrieve the content text for prediction
                test_text = row['message_detail']

                # Tokenize and create input tensors for the test text
                test_encoding = tokenizer.encode_plus(test_text, truncation=True, padding=True, return_tensors="pt")
                test_input_ids = test_encoding["input_ids"].to(device)
                test_attention_mask = test_encoding["attention_mask"].to(device)

                # Use the trained model to make a prediction on the test text
                model.eval()
                with torch.no_grad():
                    inputs = {"input_ids": test_input_ids, "attention_mask": test_attention_mask}
                    outputs = model(**inputs)
                    logits = outputs.logits
                    prediction = torch.argmax(logits, axis=1).item()

                # Update the label in the test data with the predicted value
                test_data.at[i, 'label'] = prediction

        # Save the updated test data to a CSV file separated by ';'
        test_data.to_csv("static/csv/twitter.csv", index=False, sep=';')

        return 'Labeled tweets added to twitter.csv'
    else:
        return 'All tweets are already labeled'

@app.route('/predicttlg', methods=['POST', 'GET'])
def predicttlg():
    # Load the test set and create input tensors
    # Assume that the test data is in CSV format with columns: 'message_detail', 'username', and 'content'
    test_data = pd.read_csv('static/csv/telegram.csv', sep=';', encoding='utf-8')

    # Filter out the rows with 'nan' content
    test_data = test_data.dropna(subset=['content'])

    if len(test_data) > 0:
        # Iterate over each row in the test data
        for i, row in test_data.iterrows():
            # Check if the label is NaN
            if pd.isnull(row['label']):
                # Retrieve the content text for prediction
                test_text = row['content']

                # Tokenize and create input tensors for the test text
                test_encoding = tokenizer.encode_plus(test_text, truncation=True, padding=True, return_tensors="pt")
                test_input_ids = test_encoding["input_ids"].to(device)
                test_attention_mask = test_encoding["attention_mask"].to(device)

                # Use the trained model to make a prediction on the test text
                model.eval()
                with torch.no_grad():
                    inputs = {"input_ids": test_input_ids, "attention_mask": test_attention_mask}
                    outputs = model(**inputs)
                    logits = outputs.logits
                    prediction = torch.argmax(logits, axis=1).item()

                # Update the label in the test data with the predicted value
                test_data.at[i, 'label'] = prediction

        # Save the updated test data to a CSV file separated by ';'
        test_data.to_csv("static/csv/telegram.csv", index=False, sep=';')

        return 'Labeled content is added to telegram.csv'
    else:
        return 'All content is already labeled'

@app.route('/predictinv', methods=['POST', 'GET'])
def predictinv():
    # Load the test set and create input tensors
    # Assume that the test data is in CSV format with columns: 'message_detail', 'username', and 'content'
    test_data = pd.read_csv('static/csv/investing.csv', sep=';', encoding='utf-8')

    # Filter out the rows with 'nan' content
    test_data = test_data.dropna(subset=['content'])

    if len(test_data) > 0:
        # Iterate over each row in the test data
        for i, row in test_data.iterrows():
            # Check if the label is NaN
            if pd.isnull(row['label']):
                # Retrieve the content text for prediction
                test_text = row['content']

                # Tokenize and create input tensors for the test text
                test_encoding = tokenizer.encode_plus(test_text, truncation=True, padding=True, return_tensors="pt")
                test_input_ids = test_encoding["input_ids"].to(device)
                test_attention_mask = test_encoding["attention_mask"].to(device)

                # Use the trained model to make a prediction on the test text
                model.eval()
                with torch.no_grad():
                    inputs = {"input_ids": test_input_ids, "attention_mask": test_attention_mask}
                    outputs = model(**inputs)
                    logits = outputs.logits
                    prediction = torch.argmax(logits, axis=1).item()

                # Update the label in the test data with the predicted value
                test_data.at[i, 'label'] = prediction

        # Save the updated test data to a CSV file separated by ';'
        test_data.to_csv("static/csv/investing.csv", index=False, sep=';')

        return 'Labeled content is added to investing.csv'
    else:
        return 'All content is already labeled'

if __name__ == '__main__':
    app.run(host='0.0.0.0')

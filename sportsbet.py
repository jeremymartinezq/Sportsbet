import requests
import pandas as pd
import logging
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from celery import Celery
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
import time

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Constants
DATABASE_URI = 'sqlite:///sports_betting.db'
API_KEY = 'your_api_key'
API_ODDS_URL = 'https://api.theoddsapi.com/v4/sports/odds'
API_HISTORICAL_URL = 'https://api.sportradar.com/sportsdata'


# Data Acquisition
def fetch_data(api_url, api_key, params=None):
    try:
        response = requests.get(api_url, headers={'Authorization': f'Bearer {api_key}'}, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f'HTTP error occurred: {http_err}')
    except requests.exceptions.RequestException as err:
        logging.error(f'Request error occurred: {err}')
    return None


def fetch_real_time_odds(api_url, api_key):
    data = fetch_data(api_url, api_key)
    if data:
        return pd.DataFrame(data.get('odds', []))
    return pd.DataFrame()


def fetch_historical_data(api_url, api_key, sport, start_date, end_date):
    params = {'start_date': start_date, 'end_date': end_date}
    data = fetch_data(f"{api_url}/{sport}", api_key, params)
    if data:
        return pd.DataFrame(data.get('historical', []))
    return pd.DataFrame()


# Data Storage
Base = declarative_base()
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()


class RealTimeOdds(Base):
    __tablename__ = 'real_time_odds'
    id = Column(Integer, primary_key=True)
    data = Column(String)


Base.metadata.create_all(engine)


def save_data_to_db(df, table_name):
    try:
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    except SQLAlchemyError as e:
        logging.error(f"Error saving data to DB: {e}")


def retrieve_data_from_db(table_name):
    try:
        with engine.connect() as conn:
            return pd.read_sql_table(table_name, conn)
    except SQLAlchemyError as e:
        logging.error(f"Error retrieving data from DB: {e}")
        return pd.DataFrame()


# Predictive Modeling
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(X_train, y_train, X_val, y_val):
    model = build_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=1)
    return model, history


def predict_with_model(model, X_test):
    return model.predict(X_test)


# Data Visualization
def plot_historical_odds(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['odds'], mode='lines+markers', name='Odds'))
    fig.update_layout(title='Historical Odds', xaxis_title='Date', yaxis_title='Odds')
    fig.show()


# Web Interface
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    odds_df = retrieve_data_from_db('real_time_odds')
    return render_template('index.html', tables=[odds_df.to_html(classes='data')], titles=odds_df.columns.values)


# Scheduling and Asynchronous Tasks
celery_app = Celery('tasks', broker='redis://localhost:6379/0')


@celery_app.task
def fetch_and_store_data():
    logging.info('Fetching and updating data...')
    odds_df = fetch_real_time_odds(API_ODDS_URL, API_KEY)
    save_data_to_db(odds_df, 'real_time_odds')
    logging.info('Data updated successfully.')


# API Endpoints
fastapi_app = FastAPI()


class Query(BaseModel):
    sport: str
    start_date: str
    end_date: str


@fastapi_app.get("/historical_data")
def get_historical_data(query: Query):
    data = fetch_historical_data(API_HISTORICAL_URL, API_KEY, query.sport, query.start_date, query.end_date)
    if data.empty:
        raise HTTPException(status_code=404, detail="Data not found")
    return data.to_dict()


# Run Flask and FastAPI applications
def run_flask():
    app.run(debug=True, use_reloader=False)


def run_fastapi():
    uvicorn.run(fastapi_app, host='0.0.0.0', port=8000)


if __name__ == '__main__':
    threading.Thread(target=run_flask).start()
    threading.Thread(target=run_fastapi).start()

    # Keep the script running
    while True:
        time.sleep(60)



from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import pickle
import numpy as np
import requests
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model for database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'user' or 'admin'

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Login page route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    
    return render_template('login.html')

# Register page route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        role = request.form['role']
        user = User(username=username, email=email, password=hashed_password, role=role)
        db.session.add(user)
        db.session.commit()

        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

# User Dashboard page route
@app.route('/user_dashboard')
@login_required
def user_dashboard():
    return render_template('dashboard.html', username=current_user.username)

# Admin Dashboard page route
@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    return render_template('admin_dashboard.html', username=current_user.username)

# Disaster Prediction Page
@app.route('/disaster_prediction')
def disaster_prediction():
    return render_template('disaster_prediction.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Load user function for login_manager
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the forest fire model
forest_fire_model_path = os.path.join("models", "forest_fire_model.pkl")
with open(forest_fire_model_path, "rb") as model_file:
    forest_fire_model = pickle.load(model_file)

# Initialize the scaler for forest fire model
forest_fire_scaler = StandardScaler()
forest_fire_scaler.fit([[21, 30, 70]])  # Example data

@app.route("/forest_fire", methods=["GET", "POST"])
def forest_fire():
    result = None
    if request.method == "POST":
        try:
            oxygen = float(request.form["oxygen"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])

            input_values = np.array([[oxygen, temperature, humidity]])
            scaled_data = forest_fire_scaler.transform(input_values)

            prediction = forest_fire_model.predict(scaled_data)[0]
            result = "Fire will occur." if prediction == 1 else "No fire."
        except Exception as e:
            result = f"Error occurred: {e}"

    return render_template("forest_fire.html", result=result)

# Load the flood prediction model
flood_model_path = os.path.join("models", "flood_prediction_model.pkl")
flood_model = joblib.load(flood_model_path)

# Function to fetch weather data for flood prediction
def fetch_weather_data(city, api_key="110ce3993039429987f83819252601"):
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "city_name": data['location']['name'],
            "country": data['location']['country'],
            "temp_c": data['current']['temp_c'],
            "humidity": data['current']['humidity'],
            "precip_mm": data['current']['precip_mm'],
            "wind_kph": data['current']['wind_kph'],
            "pressure_mb": data['current']['pressure_mb'],
            "condition": data['current']['condition']['text'],
        }
    else:
        return None

# Route for the flood prediction page
@app.route('/flood')
def flood():
    return render_template('flood.html')

@app.route('/predict', methods=['POST'])
def predict():
    city = request.form['city']
    weather_data = fetch_weather_data(city)
    
    if not weather_data:
        return jsonify({"error": "Enter a valid city name!"}), 400
    
    # Extract features for prediction
    features = [
        weather_data['temp_c'],
        weather_data['humidity'],
        weather_data['precip_mm'],
        weather_data['wind_kph'],
        weather_data['pressure_mb'],
    ]
    
    # Convert features to NumPy array
    input_features = np.array([features])
    
    # Make a prediction
    prediction = flood_model.predict(input_features)
    result = "Flood" if prediction[0] == 1 else "No Flood"
    
    weather_data['result'] = result  # Include the prediction in the response
    return jsonify(weather_data)


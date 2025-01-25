from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import pickle
import numpy as np
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

# Initialize the database (run this once to create the tables)
# @app.before_first_request
# def create_tables():
#     db.create_all()

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
        role = request.form['role']  # Default role is user, you can change it as needed
        print(role)
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

model_path = os.path.join("models", "forest_fire_model.pkl")
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Initialize the scaler (as before)
scaler = StandardScaler()
sample_data = np.array([[21, 30, 70]])  # Example: Oxygen, Temperature, Humidity
scaler.fit([sample_data[0]])

@app.route("/forest_fire", methods=["GET", "POST"])
def forest_fire():
    result = None
    if request.method == "POST":
        try:
            oxygen = float(request.form["oxygen"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])

            input_values = np.array([[oxygen, temperature, humidity]])
            scaled_data = scaler.transform(input_values)

            prediction = model.predict(scaled_data)[0]
            result = "Fire will occur." if prediction == 1 else "No fire."
        except Exception as e:
            result = f"Error occurred: {e}"

    return render_template("forest_fire.html", result=result)

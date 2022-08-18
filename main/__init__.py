from flask import Flask

app = Flask(__name__)

app.secret_key = "120201"

from main import routes
from flask import Flask
from .app-api-flask import app

#Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')

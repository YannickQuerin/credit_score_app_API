from flask import Flask
from .app_API import app
#from . import models

#Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')
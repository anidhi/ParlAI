from flask import Flask
from flask_ask import Ask, statement, question, session
import logging
import os
import random
import subprocess

app = Flask(__name__)
ask = Ask(app, "/")

@ask.intent("AskDeepPavlov", mapping={'user_input':'raw_input'})
def response_from_model():
	# Don't want this to be run every single time. 
	cmd = 'python3 examples/interactive.py -m hred/hred -mf test_hred.checkpoint.checkpoint'
    os.system(cmd)

	response = subprocess.check_output('user_input').decode()
	return question(response)

if __name__ == '__main__':
	app.run()

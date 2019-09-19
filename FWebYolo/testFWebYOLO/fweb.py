# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/9/16 10:47
# Name:         fweb
# Description:

from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/hello')
def hello():
    return 'hi, world!'

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % username

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        do_the_login()
    else:
        show_the_login_form()

def do_the_login():
    print("heihei")
def show_the_login_form():
    print("heeh")

if __name__ == '__main__':
    app.run(debug=True)
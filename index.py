from flask import Flask, render_template, request
# from flask_lt import run_with_lt
from a2web import Query
import os
import time

Q = Query()

app = Flask(__name__)
# run_with_lt(app)


last_time = {}


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        result = None
        query = None
        model = None
        length = None

    if request.method == 'POST':
        client_ip = request.remote_addr
        current_time = int(time.time())
        time_difference = current_time - last_time.get(client_ip, 0)
        if time_difference < 5:
            return 'Form submission rate exceeded.'
        else:
            last_time[client_ip] = current_time
            data = request.form
            query = data.get("query")
            model = data.get("model")
            print(query, model)
            result = Q.search(query, model)
            length = len(result["result"])

    return render_template('search.html', result=result, query=query, model=model, length=length)


# @app.route('/test', methods=['GET'])
# def test():
#     result = {'result': [{'rank': 1, 'doc_id': '148375', 'title': 'James Taylor', 'score': 6.476306652521549, 'topic': ['Biography/WikiProject Musicians'],
#                           'summary': 'james taylor american singersongwriter guitarist article solo musician kool gang member ... 100 million records worldwide taylor achieved breakthrough 1970 3 single fire rain 1 hit 1971 recording youve friend written carole king ... '}], 'time': 47.617645263671875}
#     return render_template('test.html', result=result, query="japan", model="all")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("FLASK_RUN_PORT", "80")))

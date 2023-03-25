from flask import Flask, request, render_template, url_for

#Flask 객체 인스턴스 생성
app = Flask(__name__)

@app.route('/') #접속하는 url
def index(num=None):
    return render_template('index.html')

@app.route('/calculate', methods=['POST', 'GET'])
def calculate(num=None):
    if request.method == 'POST':
        temp = request.form['num']
        temp = int(temp)

        temp1 = request.form['char1']
        ## 넘겨받은 값은 원래 페이지로 리다이렉트
        return render_template('index.html', num=top, char1=temp1)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

from flask import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app= Flask(__name__)

@app.route("/")
def pics():
	return render_template("pics.html")

	
@app.route("/home",methods=["GET","POST"])
def home():
	return render_template("home.html")

@app.route("/code",methods=["GET","POST"])
def code():
	if request.method=="POST":
		b=int(request.form['battery_power'])
		bl=int(request.form['blue'])
		cs=float(request.form['clock_speed'])
		ds=int(request.form['dual_sim'])
		fc=int(request.form['fc'])
		t=int(request.form['four_g'])
		im=int(request.form['int_memory'])
		md=float(request.form['m_dep'])
		mw=int(request.form['mobile_wt'])
		p=int(request.form['n_cores'])
		pc=int(request.form['pc'])
		pr=int(request.form['px_height'])
		prw=int(request.form['px_width'])
		ram=int(request.form['ram'])
		sh=int(request.form['sc_h'])
		sw=int(request.form['sc_w'])
		tt=int(request.form['talk_time'])
		g=int(request.form['three_g'])
		ts=int(request.form['touch_screen'])
		wifi=int(request.form['wifi'])
		#pf=int(request.form['pr'])
		#print()
		#data = pd.read_csv(r'D:\Mobile Price Prediction\Frontend')
		#print(data.head())
		path="mb.csv"
		#names=['b','bl','cs','ds','fc','t','im','md','mw','p','pc','pr','prw','ram','sh','sw','tt','g''ts','wifi']
		
		data=pd.read_csv(path)
		#print([b,bl,type(cs),ds,fc,t,im,md,mw,p,pc,pr,prw,ram,sh,sw,tt,g,ts,wifi])

		#return "hai"
		#sns.set()

		#plt.figure(figsize=(12, 10))
		#sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linecolor='white', linewidths=1)
		#plt.show()
		x = data.iloc[:, :-1].values
		y = data.iloc[:, -1].values
		x = StandardScaler().fit_transform(x)
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

		lreg = LogisticRegression()
		lreg.fit(x_train, y_train)
		y_pred = lreg.predict(x_test)
		accuracy = accuracy_score(y_test, y_pred) * 100
		test=lreg.predict([[b,bl,cs,ds,fc,t,im,md,mw,p,pc,pr,prw,ram,sh,sw,tt,g,ts,wifi]])
		print(accuracy,test)
		return render_template("output.html",acc=accuracy, test=test)


@app.route("/about")
def layout():
	return render_template("project.html")

@app.route("/project")
def pj():
	return render_template("about.html")

@app.route("/con")
def cont():
	return render_template("c.html")

if __name__=="__main__":
	app.run(debug=True)
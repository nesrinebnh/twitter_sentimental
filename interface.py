import tkinter as tk
from tkinter import * 
import numpy as np
import matplotlib as plt
import pandas as pd
import re

import json
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

pd.set_option('display.max_columns', None)

df=pd.read_csv("train.csv",error_bad_lines=False,encoding="ISO-8859-1")
hp=df[df['Sentiment']==1]["SentimentText"]
sd=df[df['Sentiment']==0]["SentimentText"]

happy=[]
sad=[]

for index, line in sd.items():
	line = re.sub(r'http\S+', '', line)
	line = re.sub(r'@\S+', '', line)
	line = re.sub('\W+',' ', line )
	line=line.replace("#","")
	line=line.replace("\n","")
	for i in range(10):
		line=line.replace(str(i),'' )
	sad.append(line.lower())


for index, line in hp.items():
	line = re.sub(r'http\S+', '', line)
	line = re.sub(r'@\S+', '', line)
	line = re.sub('\W+',' ', line )
	line=line.replace("#","")
	line=line.replace("\n","")
	for i in range(10):
		line=line.replace(str(i),'' )
	happy.append(line.lower())


data = happy+sad

from sklearn.model_selection import train_test_split
vec=TfidfVectorizer(stop_words='english')
X=vec.fit_transform(data)
Y=np.append(np.zeros(len(happy)),np.ones(len(sad)))
xtr,xts,ytr,yts=train_test_split(X,Y,test_size=0.33)


from sklearn.linear_model import LogisticRegression
bnb=LogisticRegression(solver='lbfgs')
bnb.fit(xtr,ytr)



def hello(sentence):
	vect=vec.transform([sentence.lower()]).toarray()
	r=bnb.predict_proba(vect)
	a = {'happiness':r[0][0], 'sadness':r[0][1] }
	return a
	



def show_entry_fields():
	Sentiment = hello(year_input.get())
	print(type(Sentiment.get('happiness')))
	if(Sentiment.get('happiness')> Sentiment.get('sadness')):

		label_day = Label(frame, text=year_input.get()+" is a happy sentence",font=("Arial",14),bg='#619ecb')
		label_day.pack()
	else:
		label_day = Label(frame, text=year_input.get()+" is a sad sentence. Are you ok?",font=("Arial",14),bg='#619ecb')
		label_day.pack()

	
	year_input.delete(0, tk.END)


# creer la fenetre
window = Tk()
#personnaliser la fenetre
window.title("Are you positive or negative?")
window.config(background='#ffffff')
window.geometry("600x600")
frame = Frame(window, bg='#ffffff')


#creer un titre
label_year = Label(frame, text="Give a comment (positive or negative)",font=("Arial",14),bg='#ffffff')
label_year.pack()

#creer une entrée/input
year_input = Entry(frame,font=("Arial",20),bg='#ffffff')
year_input.pack()

button = Button(frame, text="predict",font=("Arial",14), bg='#ff8000', bd=0, relief=SUNKEN, command=show_entry_fields)
button.pack(pady=(10,0))

#centrer le frame
frame.pack(expand=YES)
# affichage
window.mainloop()

all:app.py
	pyinstaller -F app.spec
	mv dist/app Fantastic-Filter/opt/FantasticFilter/
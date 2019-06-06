all:app.py
	pyinstaller -F app.spec
	mv dist/app Fantastic-Filter/opt/FantasticFilter/
	dpkg -b Fantastic-Filter/

install:
	sudo dpkg -i Fantastic-Filter.deb

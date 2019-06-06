all:app.py
	pyinstaller -F app.spec
	mv dist/app Fantastic-Filter.app/Contents/MacOS/
	dpkg -b Fantastic-Filter/


install:
	sudo cp -r Fantastic-Filter.app /Applications/

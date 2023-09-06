default : ui/hw1_ui.py
	python3 hw1.py

dl : ui/dl_ui.py
	python3 dl.py

hw1_ui.py : ui/hw1_ui.ui
	pyuic5 -x ui/hw1_ui.ui -o ui/hw1_ui.py

dl_ui.py: ui/dl_ui.ui
	pyuic5 -x ui/dl_ui.ui -o ui/dl_ui.py
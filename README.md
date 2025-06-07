## How to run the code

NOTE: Python version required - "3.11"


once installed - run 
- python3.11 -m venv py311
- source py311/bin/activate
- pip install -r requirements.txt


NOTE: Python should be arm64 build otherwise the required tensorflow version will not show up in pip.

1. python process_s3_images.py
Requirements:
	- Create file "aws_key_code.txt"
	- in first line write your key
	- in second line write your code
	- save it in aol-facetracker folder



2. python main.py

URL for report - http://0.0.0.0:8000/report/2025-04-27/2025-05-04


URL for facereport - http://0.0.0.0:8000/facereport/25




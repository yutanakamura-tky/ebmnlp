# 1. Preparation
1. Install necessary packages:
```sh
$ pip install -r requirements.txt
```

2. Download trained model (checkpoint) from Google Drive. [This is the link.](https://drive.google.com/drive/folders/1cID_mPxRnq6nQ6ZmbL0S3wyGslD--JsT?usp=sharing)  
Instead, you can also download it by running this:
```sh
```

# 2. Use via command line
1. Prepare text file that contains an RCT abstract (e.g., sample.txt).

2. Run like this:
```sh
$ python ebmnlp.py TEXT_FILE_NAME
```

3. NER tagging result will be returned in a standard output:
```
I-I	Remdesivir
O	in
I-P	adults
I-P	with
I-P	severe
I-P	COVID-19
I-P	:
O	a
O	randomised
O	,
O	double-blind
O	,
O	placebo-controlled
O	,
O	multicentre
```

4. If you wish to get the result as a file, run like this:
```sh
$ python ebmnlp.py TEXT_FILE_NAME OUTPUT_FILE_NAME
```

# 3. Use via Web browser
1. Run this:
```
$ bash run_flask. sh
```

2. Access to `localhost:5000` via your Web browser.

# DS test task

### Create virtualenv
```
pip3 install virtualenv
virtualenv test_env --python=python3.7
```
Mac OS / Linux
```
source test_env/bin/activate
```
Windows
```
test_env\Scripts\activate
```
### Install requirements
```
pip3 install -r requirements.txt
```
### Run module
First, add `train.tsv` and `test.tsv` into folder `data`

Then run command
```
python3 module/Preprocessor.py --train_filename="data/train.tsv" --test_filename="data/test.tsv" --output_folder="result" --output_filename="test_proc.tsv" --process_method="standardization"
```

git clone https://github.com/nubificus/python-vaccel
cd python-vaccel
python3 builder.py
pip3 install build
python3 -m build
pip install dist/vaccel*.tar.gz
export VACCEL_BACKENDS=/usr/local/lib/libvaccel-noop.so 
export LD_LIBRARY_PATH=/usr/local/lib 
export PYTHONPATH=$PYTHONPATH:. 
pytest
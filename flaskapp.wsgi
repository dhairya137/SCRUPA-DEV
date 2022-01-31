#flaskapp.wsgi
import sys
sys.path.insert(0,"/home/ubuntu/huawei-knowledge-graphs/PD/mustafa/lib/python3.8/site-packages")
sys.path.insert(0, '/home/ubuntu/huawei-knowledge-graphs/PD/')

from flaskapp import app as application

# create by andy at 2022/5/11
# reference: 
from __future__ import absolute_import
import os
from celery import Celery
from django.conf import settings
import time

# set the default Django settings module for the 'celery' program.
# from deep_models.MyFCN.train import run
from dl.train import train
from new_deep_models.deep_learning.train import new_run

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_gd.settings')
app = Celery('django_gd')

# Using a string here means the worker will not have to
# pickle the object when using Windows.
app.config_from_object('django.conf:settings')
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)


@app.task(bind=False)
def my_sum(a, b):
    time.sleep(4)
    print("------------------my_sum----------------------")
    print(a + b)
    return a + b


@app.task(bind=False)
def run_my_model(**kwargs):
    p = {}
    for key in kwargs:
        p.update({key: kwargs[key][0]})
    # run(**p)
    return kwargs


@app.task(bind=False)
def run_new_model(**kwargs):
    # new_run()
    kwargs.pop("id")
    print(kwargs)
    new_run(**kwargs)


@app.task(bind=False)
def run_dl_model(**kwargs):
    kwargs.pop("id")
    print(kwargs)
    train(**kwargs)
if __name__ == '__main__':
    pass

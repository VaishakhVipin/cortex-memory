import redis
from .config import REDIS_CONFIG

r = redis.Redis(**REDIS_CONFIG)

def set_key(key, value):
    return r.set(key, value)

def get_key(key):
    return r.get(key)
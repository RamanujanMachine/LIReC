auto_pcf_config = {
    'depth': 10000,
    'precision': 50,
    'force_fr': True,
    'timeout_sec': 60,
    'timeout_check_freq': 1000,
    'no_exception': False
}

# If you make your own database, 'name' must match the name in 'create.sql' in the line 'CREATE DATABASE <name>'
db_config = {
    'host': 'database-1.c1keieal025m.us-east-2.rds.amazonaws.com',
    'port': 5432,
    'user': 'spectator_public',
    'passwd': 'helloworld123',
    'name': 'lirec-main'
}

def get_connection_string(db_name=None):
    conf = db_config.copy()
    if db_name:
        conf['name'] = db_name
    return 'postgresql://{user}:{passwd}@{host}:{port}/{name}'.format(**conf)

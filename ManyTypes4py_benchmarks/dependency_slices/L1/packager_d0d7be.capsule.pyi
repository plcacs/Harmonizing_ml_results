# === Internal dependency: chalice.compat ===
def pip_import_string(): ...

# === Internal dependency: chalice.constants ===
MISSING_DEPENDENCIES_TEMPLATE = '\nCould not install dependencies:\n%s\nYou will have to build these yourself and vendor them in\nthe chalice vendor folder.\n\nYour deployment will continue but may not work correctly\nif missing dependencies are not present. For more information:\nhttp://aws.github.io/chalice/topics/packaging.html\n\n'

# === Internal dependency: chalice.utils ===
class OSUtils(object):
    ...
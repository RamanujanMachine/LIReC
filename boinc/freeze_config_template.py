"""
This file is the template build configuration for cx_Freeze to generate an executable binary across platforms
Parameters EXECUTABLE and APP_DESCRIPTION come from the GitHub workflow
"""
from cx_Freeze import setup, Executable

build_options = {'packages': [],
                 'excludes': ['tkinter'],
                 'build_exe': 'build/executable' # sets name of folder under build folder in which executables end up
                 }

base = 'console'

executables = [
    Executable('execute_from_json.py', base='console', target_name='{EXECUTABLE}')
]

setup(name='{EXECUTABLE}',
      version='0.1',
      description='{APP_DESCRIPTION}',
      options={'build_exe': build_options},
      executables=executables)

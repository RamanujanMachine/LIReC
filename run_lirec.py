import subprocess
import sys

def install_system_dependencies():
    # Install system packages
    subprocess.check_call("sudo yum -y update", shell=True)
    subprocess.check_call("sudo yum -y groupinstall 'Development Tools'", shell=True)
    subprocess.check_call("sudo yum -y install openssl-devel bzip2-devel libffi-devel postgresql-devel", shell=True)

def install_python():
    # Download and install Python 3.8.10
    subprocess.check_call("wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz", shell=True)
    subprocess.check_call("tar xvf Python-3.8.10.tgz", shell=True)
    subprocess.check_call("cd Python-3.8.10 && ./configure --enable-optimizations && sudo make altinstall", shell=True)

def install_lirec():
    # Install LIReC from GitHub
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/RamanujanMachine/LIReC.git"])

def run_lirec():
    # Assuming LIReC has an entry point in its package
    from LIReC import main
    main.run()

def main():
    install_system_dependencies()
    install_python()
    install_lirec()
    run_lirec()

if __name__ == "__main__":
    main()

apt-get update
apt-get install mc

# if php and apache not installed
apt-get install apache2
apt-get install php libapache2-mod-php

# default directory
cd /var/www/html

# or unzip archive
git clone https://github.com/zab88/building_profile.git

# if python3 not installed
apt-get install python3.6

apt-get install python3-pip
apt-get install build-essential python3-dev
pip3 install -r requirements.txt

pip3 uninstall opencv_contrib_python
apt-get install python3-opencv

chmod 777 uploads

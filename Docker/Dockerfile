FROM opentable/anaconda

RUN apt-get update
RUN apt-get install -y git python-qt4 build-essential python-mysqldb libmysqlclient-dev

RUN pip install seaborn yahmm pythonic-porin MySQL-python

WORKDIR /home
RUN git clone https://github.com/jvivian/Helicase-Reread-Project
WORKDIR /home/Helicase-Reread-Project/Docker
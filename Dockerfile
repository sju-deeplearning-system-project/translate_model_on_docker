FROM python:3.7-buster

ADD . /letr
WORKDIR /letr

RUN apt-get update && apt-get install -y --no-install-recommends tzdata g++ git curl
RUN apt-get install -y default-jdk default-jre

RUN pip install --upgrade pip

# for mecab-ko-dic : for command ldconfig
ENV PATH ${PATH}:/usr/local/sbin:/usr/sbin:/sbin

RUN pip install --no-cache-dir -r requirements.opt.txt --upgrade --force-reinstall
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -s

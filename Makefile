install:
	wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz -O ldcc-20140209.tar.gz
	tar -xvf ldcc-20140209.tar.gz
	git clone https://github.com/PrincetonML/SIF.git
	pip install -r SIF/requirements.txt
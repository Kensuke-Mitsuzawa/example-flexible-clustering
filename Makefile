install:
	wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz -O ldcc-20140209.tar.gz
	tar -xvf ldcc-20140209.tar.gz
	git clone https://github.com/PrincetonML/SIF.git
	pip install -r SIF/requirements.txt
	wget https://github.com/singletongue/WikiEntVec/releases/download/20190520/jawiki.all_vectors.100d.txt.bz2 -O text/jawiki.all_vectors.100d.txt.bz2
	bunzip2 -dc text/jawiki.all_vectors.100d.txt.bz2 > text/jawiki.all_vectors.100d.txt

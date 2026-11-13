all:
	mkdir -p public
	rsync -a src/ public/

PROJECT_NAME=transformer

USER=vs
REMOTE_SERVER=adrian_minimal
REMOTE_FOLDER=transformer


.PHONY: install train evaluate notebook remote-sync sync-remote

install:
	@pip install -r requirements.txt

train:
	python ct.py train

evaluate:
	@echo "Not implemented"

notebook:
	jupyter notebook --port=7888

remote-sync:
	@rsync -avze ssh \
		--exclude 'data/*' \
		--exclude '*.ipynb' \
		--exclude '.git/' \
		--exclude '.idea/' \
		--exclude '*.pyc' \
		--exclude 'docs/build/*' \
		--include '*.py' \
		--progress \
		. \
		$(USER)@$(REMOTE_SERVER):~/$(REMOTE_FOLDER)

sync-remote: remote-sync
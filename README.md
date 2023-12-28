# RIR_estimator

This repository presents a deep learning approach to the automatic estimation of absorption coefficients as well as room geometry (shoebox model) from room-impulse-responses (RIR).

The system is trained using virtually generated RIR .wav files. These files are created through the "create_corpus.py" file, using pyroomacoustics. The resulting generated files are then split into train, validation and test manifests.

The full parameter list for both the corpus creation and system training can be found in the ./configs/parameters.yaml file.

Repository structure:

	> configs/
		- parameters.yaml
	> data/
		> /manifests/
		> /rir_clips/
	> dataloader/
		- load.py
	> models/
		- model.py
	> resources/
		- materials_index_pra.csv
		- burst_balloon.wav
	- README.md
	- .gitignore
	- requirements.txt
	- create_corpus.py
	- train.py




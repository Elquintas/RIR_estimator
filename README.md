# RIR_estimator

Author: Sebastião Quintas

This repository presents a deep learning approach to the automatic estimation of absorption coefficients as well as room geometry (shoebox model) from room-impulse-responses (RIR). The system is trained using virtually generated RIR files (wav), using pyroomacoustics to simulate different shoebox rooms with varying absorption levels.

The main recipe can be found divided in two parts. The first corresponds to the corpus creation: 
	
	$ python create_corpus.py

This script creates a corpus based on the virtual room images generated by pyroomacoustics. Each RIR file is generated using a random room geometry, where for each surface, 4 distinct walls, floor and ceiling, a random material is sampled. The list (materials_index_pra.csv) contains 72 different materials obtained from pyroomacoustics website, with the absorption coefficients for the first six bands [0.125, 0.250. 0.500. 1.0, 2.0 and 4.0 kHz]. This creates a large number of possible room combinations with varying sizes as well as material combinations (72^6).

The microphone and source are added randomly to the room following the criteria that they're at least 0.5m appart from any surface and at least 1m appart from each other. The RIR files are either padded or truncated to fit a 0.5s window (fs/2 = 8000Hz). Files are corrupted with additive white noise with a SNR of 30db. Ray tracing as well as a max order of 10 reflections are used to generate files.


The second part of the global recipe corresponds to the train/test procedure:
	
	$ python train.py

This script uses the corpus generated previously and separates it into the different manifests (train, val and test). The dataloaders can be found described in the load.py file, while the model and lightning module can be found on the model.py script. Two classes of systems are proposed, both based on CNNs. The first one predicts only the weighted mean absorption coefficients, while the second predicts the coefficients in tandem with the room geometry (width, length and height), using a multi-task learning methodology.

Results are stored in a .csv file under the results directory. Inferences can be run using the following command

	$ python predict.py path/to/rir.wav

This script predicts the absorption coefficients (and room geometry, if multi-task) given a specific checkpoint and rir file.

Progress can be tracked using tensorboard:

	$ tensorboard --logdir=lightning_logs/

The full parameter list for both the corpus creation and system training can be found in the ./configs/parameters.yaml file. Helper functions can be found under the resources directory (e.g. plot_rir.py to plot the waveform of a given room-impulse-response file).

Repository structure:

	> /configs/
		- parameters.yaml
	> /data/
		> /manifests/
			...
		> /rir_clips/
			...
	> /dataloader/
		- load.py
	> /models/
		- model.py
	> /utils/
		- materials_index_pra.csv
		- plot_rir.py
		- results_analysis.py
	> /results/
	> /pre-trained/

	- .gitignore
	- README.md
	- requirements.txt
	- create_corpus.py
	- train.py
	- predict.py



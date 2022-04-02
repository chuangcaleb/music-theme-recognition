# Classification of Themes and Motifs in Musical Composition with MIDI

A multi-label classifier algorithm to predict motifs/themes in musical composition.

- [About the Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [License](#license)
- [Contact](#contact)

## About the Project

This project is for my individual dissertation of my Bachelor's.

## Built With

- [Python](https://www.python.org/)

## Getting Started

### Prerequisites

- matplotlib
- scipy
- numpy
- jupyter
- scikit-learn
<!-- - scikit-multilearn[^1]

[^1]: There was a major issue with the MLkNN class, see [here](https://github.com/scikit-multilearn/scikit-multilearn/issues/224#) on how to fix it. -->

Python related dependencies can be installed using:

```python
pip install -r requirements.txt
```

### Installation

Clone the repo:

```sh
git clone https://github.com/chuangcaleb/music-theme-recognition
```

## Usage

### 1. [collecting_data](collecting_data/)

There are two stages to collecting the required dataset.

#### 1a. [scraping_midi](collecting_data/1_scraping_midi)

Different scripts download MIDI files from various sources into [a bin directory](data/bin/). Manually downloaded MIDI files can be manually added in here as well.

#### 1b. [building_dataset](collecting_data/2_building_dataset)

1. `create_db.py` goes through the bin directory and builds a database containing the ids of the samples.
2. From here, I have manually added the theme labels as columns, as well as metadata columns such as `duplicate`.
3. Then, the samples are slowly labelled, marking songs that I've looked through with a 1 in the `recognizable` column if I have labelled them, and 0 if not (This means that if that field is empty/null, it has not been identified yet).
4. `process_db.py` converts all 'p's in the database into '1's.[^1]
5. `db_stats.py` is a convenience script that returns some statistics about the label dataset so far.

[^1]: This is because I have sped up the hand-labelling process by marking fields with '0' or 'p', since they are closeby on the keyboard. The script later turns the 'p's into '1's.

### 2. [calculating_dataset](calculating_dataset/)

1. `generate_jsymbolic_config.py` builds a configuration script based on the MIDI files found in the bin directory.
2. Run `jSymbolic` with [themeConfigFile.txt](calculating_dataset/themeConfigFile.txt) as the configuration script.
3. Finally, run `clean_db.py` to clean up the database for use.

These three steps can (and should) be automatically executed.

Here is a script file that I've used — modify it to point to your jSybolic2.jar.

```sh
python3 calculating_dataset/generate_jsymbolic_config.py

java -Xmx3072m -jar [PATH_TO_YOUR_JSYMBOLIC]/jSymbolic2.jar -configrun calculating_dataset/themeConfigFile.txt

python3 calculating_dataset/clean_db.py
```

### 3. [building_model](building_model)

From here, it is mostly automated.

`model.py` is the main script to run. You should never need to fiddle with it because the parameters can all be configured with `config.py`.

## Roadmap

See the [kanban](https://github.com/chuangcaleb/music-theme-recognition/projects/1?fullscreen=true) for active tasks.

## License

<!-- Distributed under the MIT License. See `LICENSE` for more information. -->

## Contact

20204134 Chuang Caleb hcycc2

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

- [Python v3.9.7](https://www.python.org/)

## Getting Started

### Prerequisites

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

The project is subdivided into four modules, to decouple the project workflow.

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
2. Run `jSymbolic` with [theme_jsymb_config.txt](data/features/theme_jsymb_config.txt) as the configuration script.
3. Finally, run `clean_db.py` to clean up the database for use. 'definition_dump/py` dumps the definition data from xml to csv.

These three steps can (and should) be automatically executed.

Here is a script file that I've used — modify it to point to your jSybolic2.jar.

```sh
python3 calculating_dataset/generate_jsymbolic_config.py

java -Xmx3072m -jar [PATH_TO_YOUR_JSYMBOLIC]/jSymbolic2.jar -configrun data/features/theme_jsymb_config.txt

python3 calculating_dataset/clean_db.py
python3 calculating_dataset/definition_dump.py
```

### 3. [building_model](building_model/)

From here, it is mostly automated.

`model.py` is the main script to run. You should never need to fiddle with it because the parameters can all be configured with `config.py`.

### 4. [evaluating_results](evaluating_results/)

Similarly to the [building_model](building_model/) module, a `config.py` file handles configurations for this module.

#### 4a. Process Results

`results_stats.py` calculates relevant statistics about the result set.

`flatten_json.py` is a temporary utility script that converts the json results into a flat table in csv.

#### 4b. Analyze Results

`plot_trees.py` draws the best Decision Trees from the specified run.

`feature_importances.py` shows a plot of the feature value distributions for each theme label for the top 10 most-important features as calculated from the best Random Forest, from the specified run.

`graph.py` plots results from the specified run.

## Roadmap

See the [kanban](https://github.com/chuangcaleb/music-theme-recognition/projects/1?fullscreen=true) for active tasks.

## License

 Distributed under the MIT License. See `LICENSE` for more information.

### Contact

20204134 Chuang Caleb hcycc2

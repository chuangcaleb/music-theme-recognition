import requests
from bs4 import BeautifulSoup
import re
import os

source = 'midiworld'
domain = "http://www.midiworld.com"
category = 'movie%20themes'  # CHANGE THIS

OUTPUT_DIR = 'data/bin/' + source + "/" + re.sub(r'%20', "-", category)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Extract metadata from the download label
def getFileData(anchor):
    text = anchor.parent.get_text()
    filtered_text = text.replace("/", " - ")  # Handle slashes in labels
    extracted_text = re.split("(.+?)(?= \()..(.+)(?=\))", filtered_text)

    if len(extracted_text) > 1:  # Handle for weird text labels
        song_title = extracted_text[1]
        song_artist = extracted_text[2]
    else:
        song_title = extracted_text[0]
        song_artist = ""
    file_data = {"song_title": song_title,
                 "song_artist": song_artist}

    return file_data


def downloadFile(anchor, filename):

    link = anchor['href']
    mid_file = requests.get(link, stream=True)
    with open(OUTPUT_DIR + '/' + filename, 'wb') as saveMidFile:
        saveMidFile.write(mid_file.content)
        print('Downloaded \"{}\" successfully.'.format(filename))


def getFileName(song_title, song_artist):
    return song_artist + '_' + song_title + '_' + source + '.mid'


# >>> ENTRY

print(f"\n\nScraping from {domain}")
print("Type Y to scrape; anything else to skip\n")

terminate = 0
page_number = 1

while terminate == 0:

    ext = f"/search/{page_number}/?q="
    url_link = domain + ext + category

    # Soup
    main_page = requests.get(url_link)
    parsed_page = BeautifulSoup(main_page.content, 'html.parser')

    anchors = parsed_page.find_all('a', href=re.compile('download'))

    if len(anchors) == 0:
        terminate = 1
        break

    for anchor in anchors:

        file_data = getFileData(anchor)

        filename = getFileName(
            file_data['song_title'], file_data['song_artist'])
        print("\n\n" + file_data['song_artist'] +
              " - " + file_data['song_title'])

        # choice = input()
        choice = "y"
        if choice.lower() == "y":
            downloadFile(anchor, filename)

    page_number = page_number + 1
    print(page_number)

print("\nEnd of scraping " + domain + " for " + category + "\n")

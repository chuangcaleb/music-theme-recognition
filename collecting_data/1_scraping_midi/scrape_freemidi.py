import webbrowser
import os
from bs4 import BeautifulSoup
from urllib import request
from urllib.request import urlopen

source = 'freemidi'
domain = "http://www." + source + ".com"

CURRENT_PATH = os.path.realpath(__file__)

print(f"\n\nScraping from {domain}")
print("Type Y to scrape; anything else to skip\n")

# Main page soup
all_page = domain + "all"
all_page_doc = request.urlopen(all_page).read().decode('utf8', errors='ignore')
main_soup = BeautifulSoup(all_page_doc, 'html.parser')

# Links
main_body = main_soup.find("div", {"class": "all-genre-con"})
category_anchors = main_body.find_all('a')

category_paths = [i["href"].replace("/", "") for i in category_anchors]
category_links = [domain + href for href in category_paths]

for url in category_links:

    # Subpage soup
    category_page = request.urlopen(url).read().decode('utf8', errors='ignore')
    category_soup = BeautifulSoup(category_page, 'html.parser')

    # Links
    category_body = category_soup.find("div", {"id": "mainContent"})
    artist_anchors = category_body.find_all('a', href=True)

    artist_paths = [[i["href"].replace("/", ""), i.get_text().replace("\n", "")]
                    for i in artist_anchors]

    for artist_path in artist_paths:
        print("\n==============\nArtist: ", artist_path[1])
        artist_choice = input()
        if artist_choice.lower() == "y":

            # Artist page soup
            artist_page = domain + artist_path[0]
            artist_doc = request.urlopen(
                artist_page).read().decode('utf8', errors='ignore')
            artist_soup = BeautifulSoup(artist_doc, 'html.parser')

            # Links
            artist_body = artist_soup.find("div", itemscope=True)
            song_anchors = artist_body.find_all('a', href=True)

            # Download links
            song_paths = [[i["href"].replace(
                "/", ""), i.get_text().replace('\n', '')] for i in song_anchors]

            # Final prompt
            for song_pair in song_paths:
                print("\nSong: ", song_pair[1])
                artist_choice = input()

                if artist_choice.lower() == "y":

                    # Download page soup
                    download_page = domain + song_pair[0]
                    download_doc = request.urlopen(
                        download_page).read().decode('utf8', errors='ignore')
                    download_soup = BeautifulSoup(download_doc, 'html.parser')

                    # Links
                    download_elem = download_soup.find(
                        "a", {"id": "downloadmidi"})
                    download_link = domain + download_elem["href"]
                    print(download_link)

                    # controller = webbrowser.get(using='chrome')
                    # with open(song_pair[1] + ".mid", 'wb') as local_file:
                    webbrowser.open(download_link)
                    # r = requests.get(download_link, allow_redirects=True)
                    # local_file.write(r.content)

                    #     # giving a name and saving it in any required format
                    #     # opening the file in write mode
                    # urllib.request.urlretrieve(download_link, "C.mid")
                    # for data in file_stream:
                    #     local_file.write(data)


# artist_links = [domain + href for href in artist_paths]

# print(artist_links[1])
# for path in artist_paths:
#     print(path)


# keywords = [keyword.get_text()
#             for keyword in soup.find_all('dt')]
# definitions = [sent_tokenize(description.get_text().replace("\n", " "))
#                for description in soup.find_all('dd')]

# python_glossary = dict(zip(keywords, definitions))

# for item in python_glossary:
#     print(item)
#     print(python_glossary[item])
#     print("\n----\n")

# pickle.dump(python_glossary, open(
#     "python_glossary.pickle", "wb"))

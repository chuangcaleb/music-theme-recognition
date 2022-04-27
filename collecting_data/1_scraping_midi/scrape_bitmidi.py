import requests
from bs4 import BeautifulSoup
import re
import os
import time

source = 'bitmidi'
domain = "http://www." + source + ".com"

OUTPUT_DIR = 'data/bin/' + source
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

terminate = 0
page_number = 0

while terminate == 0:

    print(f"\nNow on page {page_number}\n")
    ext = f"/?page={page_number}"
    list_link = domain + ext

    # Main Soup
    main_page = requests.get(list_link, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
    })
    parsed_main_page = BeautifulSoup(main_page.content, 'html.parser')

    anchors = parsed_main_page.find_all('a', href=re.compile('mid$'))

    if len(anchors) == 0:
        terminate = 1
        break

    for anchor in anchors:

        sub_page_link = domain + anchor['href']

        success = 0

        while success != 1:

            try:
                # Subpage soup
                sub_page = requests.get(sub_page_link, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
                })
                parsed_sub_page = BeautifulSoup(
                    sub_page.content, 'html.parser')

                # Anchor
                download_anchor = parsed_sub_page.find('a', download=True)
                download_ext = download_anchor['href']

                # Heading
                download_header = parsed_sub_page.find('h1')
                download_header = download_header.get_text()
                download_header = re.sub(".mid", "", download_header)

                download_link = domain + download_ext

                mid_file = requests.get(download_link, stream=True, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
                })
                filename = download_header + "_" + source + ".mid"
                with open(OUTPUT_DIR + '/' + filename, 'wb') as saveMidFile:
                    saveMidFile.write(mid_file.content)
                    print('Downloaded {} successfully.\n'.format(download_header))

                success = 1
                # Note: we add delay to the download to prevent tripping an error
                time.sleep(1.2)
            except TypeError:
                print("reattempt")
                if success == 0:
                    print(parsed_sub_page)
                    sucesss = success - 1
                pass

    page_number = page_number + 1
    time.sleep(2)

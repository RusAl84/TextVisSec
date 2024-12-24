from bs4 import BeautifulSoup
import requests
import json


def get_data(file):
    with open(file, encoding="UTF-8") as f:
        page = f.read()
    soup = BeautifulSoup(page, "html.parser")

    trs = soup.find_all("tr")
    data = []
    for tr in trs:
        tr = BeautifulSoup(str(tr), "html.parser")
        bdu = tr.find_all("td", {"class": "bdu"})
        if len(bdu) > 0:
            desc = tr.find_all("td", {"class": "desc"})
            risk = tr.find_all("td", {"class": "risk"})
            print(
                f"bdu: {bdu[0].text}  desc: {desc[0].text} risk:{risk[0].text}\n")
            line = {}
            line['bdu'] = bdu[0].text.strip()
            line['desc'] = desc[0].text.strip()
            line['risk'] = risk[0].text.strip()
            data.append(line)
    jsonstring = json.dumps(data, ensure_ascii=False)
    with open("report_data.json", "w", encoding="UTF8") as file:
        file.write(jsonstring)
    return data


if __name__ == '__main__':
    file = "ScanOval_a.html"
    data = get_data(file)
    # print(data)
    with open("bdu.txt", "w+", encoding="UTF8") as bdu_file:
        for item in data:
            desc = item['desc']
            desc = desc.replace('\t', '').replace('\n', '').replace('\r', '')
            bdu_file.write(f"{item['bdu']}\t{desc}\n")
    # for item in data:
    #     print(f"bdu:{item['bdu']}  desc:{item['desc']}")
